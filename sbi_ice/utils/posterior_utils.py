from sbi_ice.utils import modelling_utils,noise_model,plotting_utils
from sbi_ice.simulators import Layer_Tracing_Sim as lts
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import pickle
import io

class FCEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 20,
        num_layers: int = 2,
        num_hiddens: int = 20,
    ):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            output_dim: Dimensionality of the output.
            num_layers: Number of layers of the embedding network. (Minimum of 2).
            num_hiddens: Number of hidden units in each layer of the embedding network.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, num_hiddens), nn.ReLU()]
        # first and last layer is defined by the input and output dimension.
        # therefore the "number of hidden layeres" is num_layers-2
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(num_hiddens, num_hiddens))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hiddens, output_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Network output (batch_size, num_features).
        """
        return self.net(x)



class CPU_Unpickler(pickle.Unpickler):
    #Load inference objects saved on GPU to CPU
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#TODO: simulate posterior samples through run_sims interface
def sim_post_sample(loader,x,sample,real_layer_number,layer_mask,selection_method = "MSE"):
    """
    Find the best layer for a sample from the posterior
    Parameters:
    loader: Dataloader object containing information about the original simulations used for the sample
    x: x the sample is defined on
    sample: smb sample from posterior
    real_layer_number: The index of the layer to find the best contour for
    layer_mask: Mask containing locations where layer data was used to train the posterior model.
    selection_method: Method used to select the best layer.
    """

    #First, define load in data
    geom = lts.Geom(nx_iso=loader.x.size,ny_iso=loader.cfg.iso_hparams.ny_iso)
    x_setup,bs,ss,vxs,tmb,dQdx,dQdy = lts.init_fields_from_fname(geom,loader._setup_fname)
    bounds = loader.all_layer_xbounds(overwrite=False)[real_layer_number]

    #Potentially mask to only simulate on part of the domain if required.
    #mask = x_setup>bounds
    mask = np.ones_like(x_setup,dtype=bool)
    x_setup = x_setup[mask]
    bs = bs[mask]
    ss = ss[mask]
    vxs = vxs[mask]
    tmb = tmb[mask]
    dQdx = dQdx[mask]
    dQdy = dQdy[mask]

    #Define geometry object for simulations
    geom = lts.Geom(nx_iso=loader.x.size,ny_iso=loader.cfg.iso_hparams.ny_iso)
    #Regrid smb sample onto geom grid
    smb_ = modelling_utils.regrid(x,sample,x_setup)
    smb_regrid,bmb_regrid = lts.init_from_fields(geom,x_setup,bs,ss,vxs,tmb,dQdx,dQdy,smb=smb_)
    init_layers = loader.cfg.iso_hparams.init_layers
    scheds = loader.cfg.scheds
    #Simulate
    for key in scheds.keys():
            sched = lts.Scheduele(**scheds[key])
            geom.initialize_layers(sched,init_layers)
            lts.sim(geom,smb_regrid,bmb_regrid,sched)

    #Add noise to the simulated layers
    idxs, dsum_iso, age_iso = geom.extract_nonzero_layers()
    active_trackers = geom.extract_active_trackers()
    #return all the layers, and also the layer best matching the true layer
    true_layer = torch.Tensor(loader.real_layers[real_layer_number])
    input_layers = torch.Tensor(geom.dsum_iso[:,0,:]+geom.bs[:,:]).T
    layers_thickness = geom.dsum_iso[:,0,:].T
    heights = layers_thickness + geom.bs.flatten()
    layer_depths = geom.ss.flatten() - heights
    layer_depths = torch.from_numpy(layer_depths)
    layer_depths = torch.flip(layer_depths,dims=(0,))
    layer_xs = torch.stack([torch.tensor(geom.x) for i in range(layer_depths.shape[0])])

    #If using the advanced noise method, load in the PSD file. Currently other noise models not implemented.
    if selection_method == "MSE":
        print("Warning: using advanced_noise method instead of MSE.")
    if selection_method == "advanced_noise":
        try:
            PSD_dict = pickle.load(Path(loader._fol_path,loader._exp_path,"PSD_matched_noise.p").open("rb"))
            freq = PSD_dict["freqs"][0]
            PSD_log_mean = PSD_dict["PSD_log_diffs_means"][0]
            PSD_log_var = PSD_dict["PSD_log_diffs_vars"][0]
            base_error,depth_corr,picking_error,error = noise_model.depth_error(layer_xs,layer_depths,freq,PSD_log_mean,PSD_log_var)
        except:
            print("No PSD file found! If you are using the advanced noise method, please make sure you have run the PSD matching experiment first!")
            print("Using MSE instead of advanced noise method")
            base_error,depth_corr,picking_error,error = noise_model.depth_error(layer_xs,layer_depths)
    else:
        raise NotImplementedError("method must be one of MSE or advanced_noise")
    
    #Select the best layer and return results.
    flipped_error = torch.flip(error,dims=(0,))
    input_layers = input_layers + flipped_error
    best_contour,norm,aidx = noise_model.best_contour(true_layer,input_layers,layer_mask=layer_mask,method=selection_method)
    x_eval = loader.x[layer_mask]
    #best_contour = modelling_utils.regrid(geom.x, best_contour.numpy(), x_eval,kind="linear")
    best_age = geom.age_iso[aidx]
    return geom.dsum_iso,best_contour,norm,best_age,bmb_regrid.flatten(),active_trackers
