import hydra
import os
from omegaconf import DictConfig,OmegaConf
from pathlib import Path
import logging
import pickle
import torch
import numpy as np
from sbi_ice.loaders import ShortProfileLoader
import sbi.analysis as analysis
from sbi_ice.utils import posterior_utils,modelling_utils,plotting_utils,misc
import matplotlib.pyplot as plt
from tueplots import figsizes
color_opts = plotting_utils.setup_plots()
#Change config_name for other shelves
@hydra.main(version_base=None, config_path="../../configs/post_predictive", config_name="Ekstrom_post_pred") 
def my_app(cfg : DictConfig)->None:
    #Also need config file for the training of the posterior we want to evaluate here
    logging.basicConfig(filename="post_predictive.log",encoding="utf-8",level=logging.INFO)
    logging.info("Started Posterior Predictive")
    config_fol = cfg.posterior_config_fol
    logging.info(Path(config_fol,str(cfg.name),"config.yaml"))
    n_post_samples = cfg.n_post_samples
    n_predictive_sims = cfg.n_predictive_sims
    overwrite = cfg.overwrite_saved_sims
    posterior_config = OmegaConf.load(Path(config_fol,str(cfg.name),"config.yaml"))
    logging.info(posterior_config)
    paths = posterior_config.paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_fol = os.getcwd()
    logging.info(output_fol)

    #Create dataloader and load the inference object.
    work_folder = misc.get_project_root()
    shelf_setup_path = paths.shelf_setup_path
    shelf_folder_path = paths.shelf_folder_path
    exp_path = paths.exp_path
    logging.info("Read Config files")
    loader = ShortProfileLoader.ShortProfileLoader(Path(work_folder,shelf_setup_path),Path(work_folder,shelf_folder_path),exp_path,gt_version = posterior_config.gt_version)
    logging.info("Set up data loader")
    with open(Path(config_fol,str(cfg.name),"inference.p"), "rb") as f:
        if device.type == "cpu":
            out = posterior_utils.CPU_Unpickler(f).load()
        else:
            out = pickle.load(f)
    logging.info("Read file")
    inference = out["inference"]
    prior = out["prior"]
    layer_mask = out["layer_mask"]
    smb_mask = out["smb_mask"]
    logging.info("Loaded inference object")

    true_layer = torch.tensor(loader.real_layers[cfg.layer_idx][layer_mask]).float()
    plot_layer = loader.real_layers[cfg.layer_idx].copy()
    logging.info("Real layer made")
    posterior = inference.build_posterior(inference._neural_net.to("cpu"))
    posterior.set_default_x(true_layer)
    logging.info("Posterior object made")
    #Sample from posterior
    samples = posterior.sample((n_post_samples,))
    logging.info("Posterior samples made")
    try:
        true_smb = loader._true_smb_regrid
    except:
        true_smb = None
    logging.info("Made real smb")
    spatial_samples = samples

    #If posterior samples exist already and we are not overwriting them, load and plot. Otherwise simulate posterior samples.
    if not Path(output_fol,"post_predictive.p").exists() or overwrite:
        logging.info("Running Sims for posterior samples...")
        post_samples = samples.numpy()
        smb_samples = []
        bmb_samples = []
        best_layers = []
        norms = []
        ages = []
        tracker_samples = []
        for i in range(n_predictive_sims):
            smb_sample = post_samples[i]
            dsum_iso,best_contour,norm,age,bmb,active_trackers = posterior_utils.sim_post_sample(loader,loader.x[smb_mask],smb_sample,cfg.layer_idx,layer_mask,selection_method = posterior_config.selection_method)
            smb_samples.append(smb_sample)
            bmb_samples.append(bmb)
            best_layers.append(best_contour)
            norms.append(norm)
            ages.append(age)
            tracker_samples.append(active_trackers)
        bmb_samples = np.array(bmb_samples)
        best_layers = np.array(best_layers)
        norms = np.array(norms)
        ages = np.array(ages)
        with open("post_predictive.p", "wb") as f:
            pickle.dump(dict(bmb_samples=bmb_samples,best_layers=best_layers,norms=norms,ages=ages,active_trackers=active_trackers), f)
    else:
        logging.info("Reading posterior predictive sims from file...")
        with open("post_predictive.p", "rb") as f:
            out = pickle.load(f)
            bmb_samples = out["bmb_samples"]
            best_layers = out["best_layers"]
            norms = out["norms"]
            ages = out["ages"]
            active_trackers = out["active_trackers"]

    logging.info("Finished posterior predictive sims")
    logging.info("Loader.x shape: ")
    logging.info(loader.x.shape)

    #Load prior simulations
    contour_arrays,norm_arrays,age_arrays,smb_unperturbed_all,smb_cnst_means_all,smb_sds_all,smb_all,bmb_all = loader.load_training_data(layers_fname = posterior_config.layers_fname,mb_fname = posterior_config.mb_fname)
    perm = torch.randperm(smb_all.size(0))
    idx = perm[:1000]
    prior_samples = smb_all[idx][:,smb_mask]
    prior_spatial_samples = prior_samples
    logging.info("BMB all shape: ")
    logging.info(bmb_all[0].numpy().shape)

    tmb = smb_all[0].numpy() + bmb_all[0].numpy()
    logging.info("Made tmb")
    
    #plot_layer_mask = layer_mask
    plot_layer_mask = np.ones_like(layer_mask).astype(bool)
    #Plot posterior + posterior predictive
    fig,axs = plotting_utils.plot_posterior_nice(x = loader.x,
                                            mb_mask = smb_mask,
                                            tmb = tmb,
                                            prior_smb_samples = prior_spatial_samples,
                                            posterior_smb_samples = spatial_samples,
                                            layer_mask = plot_layer_mask,
                                            LMI_boundary = loader.x[layer_mask][0],
                                            prior_layer_samples= contour_arrays[cfg.layer_idx,:n_predictive_sims,:],
                                            prior_layer_ages=age_arrays[cfg.layer_idx,:n_predictive_sims],
                                            posterior_layer_samples=best_layers,
                                            posterior_layer_ages=ages,
                                            true_layer=plot_layer,
                                            shelf_base=loader.base.flatten(),
                                            shelf_surface=loader.surface.flatten(),
                                            true_smb=true_smb,
                                            true_age=None,
                                            plot_samples=False,
                                            title=None,
                                            )

    fig.savefig("post_predictive_nice.png")
    logging.info("Done!")


if __name__ == "__main__":
    hyperparams = my_app()


