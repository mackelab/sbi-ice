from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import pandas as pd
import os
import pickle
import time
import torch
from sbi_ice.utils.modelling_utils import regrid,regrid_all

tol = 1e-3


class BaseLoader():
    """
    Base class to handle all data for a set of simulations (e.g. Ekstrom/exp2).
    This includes the inpute files to the simulations, the raw simulation outputs, the processed simulation
    outputs, and the ground truth values.

    Args:
        setup_dir (str): The directory containing the setup files (input to simulation).
        fol_path (str): The path to the folder containing the experiment files (which shelf?).
        exp_path (str): The path to the experiment files (which experiment?).
        sims_path (str, optional): The path to the layer simulation files. One of "layer_sims" or "calib_sims".
        gt_version (str, optional): The version of the ground truth mass balance and real layer elevations. Defaults to "v0".
    """

    def __init__(self,setup_dir,fol_path,exp_path,sims_path = "layer_sims",gt_version ="v0"):


        self._fol_path = fol_path
        self._exp_path = exp_path
        self._layer_sims_path = Path(self._fol_path,self._exp_path,sims_path)
        dir = Path(self._fol_path,self._exp_path)
        fname = dir.joinpath("multirun.yaml")
        self.cfg = OmegaConf.load(fname)
        self._jobs = []
        self._num_sims = []

        #Label all the jobs and the experiments in that job
        for folder in self._layer_sims_path.iterdir():

            if folder.is_dir():
                try:
                    self._jobs.append(int(folder.name))
                    f_cfg = Path(self._fol_path,self._exp_path,folder,"config.yaml")
                    cfg = OmegaConf.load(f_cfg)
                    self._num_sims.append(cfg.random_seed.nsims)
                except:
                    continue
        if len(self._jobs)==0:
            print("Warning: No jobs detected! Allowing you to continue, assuming you are only working with processed results.")
            self._jobs = [0]
            self._num_sims = [0]
        else:
            self._jobs, self._num_sims = (list(t) for t in zip(*sorted(zip(self._jobs, self._num_sims))))
        self._num_sims = np.array(self._num_sims)
        self._cum_sims = np.concatenate((np.array([0]),np.cumsum(self._num_sims)))
        self.total_sims = self._cum_sims[-1]


        #Load in the setup file of the geometry
        setup_fname = self.cfg.setup_file.address.rsplit("/", 1)[-1]
        self._setup_fname = Path(setup_dir,setup_fname)
        self._setup_df = pd.read_csv(self._setup_fname)
        self.x_coords = self._setup_df["x_coord"].to_numpy()
        self.x = np.linspace(self.x_coords[0],self.x_coords[-1],self.cfg.iso_hparams.nx_iso)
        self.base,self.surface,self.velocity,self.tmb = regrid_all(self._setup_df["x_coord"].to_numpy(),[self._setup_df["base"].to_numpy(),self._setup_df["surface"].to_numpy(),self._setup_df["velocity"].to_numpy(),self._setup_df["tmb"].to_numpy()],self.x)
        self._obs_layers_file = Path(fol_path,exp_path,"real_layers.csv")

        self.outputs_path = Path(self._fol_path,self._exp_path,"sbi_sims")
        if not self.outputs_path.is_dir():
            self.outputs_path.mkdir()

        #Load in the real layers if they exist
        if gt_version == "v0":
            self.gt_version = ""
        else:
            self.gt_version = "_"+gt_version
        layers_postfix = "real_layers" + self.gt_version + ".csv"
        self._real_layers_file = Path(self._fol_path,self._exp_path,layers_postfix)
        mb_postfix = "real_mb" + self.gt_version + ".p"
        self._real_mb_file = Path(self._fol_path,self._exp_path,mb_postfix)
        layers_supp_postfix = "real_layers" + self.gt_version + "_supp.npy"
        self._real_layers_supp_file = Path(self._fol_path,self._exp_path,layers_supp_postfix)
        try:
            self._real_layers =self.set_real_layers()
        except:
            print("No `real_layers.csv` file detected. Initialize layers using set_real_layers(filename). Cannot do inference without an observation, but you can still use this object to analyse the simulated dataset...")

        self.sim_time = self.calculate_sim_time_from_cfg()
        self.masks = []
        self._layer_bounds = []
        self._anomalies = {}

    def calculate_sim_time_from_cfg(self):
        #Calculate the total simulation time for the experiments from the config file
        #Hack for now: all sims used are 1000 years
        return 1000.0


    #Return real layers
    @property
    def real_layers(self):
        try:
            return self._real_layers
        except AttributeError:
            print("No observed layers defined! Do you have a file called `real_layers.csv` in your experiment folder?")
            raise(AttributeError)

    #Load in the real layers
    def set_real_layers(self):
        layers_df = pd.read_csv(self._real_layers_file)
        n_real_layers = len(layers_df.columns)-2
        print(n_real_layers)
        self._real_layers = np.zeros(shape=(n_real_layers,self.cfg.iso_hparams.nx_iso))
        #Regrid the real layers to the simulation grid (e.g. real data is defined on a different grid)
        for i in range(n_real_layers):
            layer_depths = regrid(layers_df["x_coord"],layers_df["layer {}".format(i+1)],self.x,kind="linear")
            self._real_layers[i,:] = self.surface-layer_depths
        try:
            self.real_layer_idxs = np.load(self._real_layers_supp_file)
        except:
            print("No supplementary information for real layers detected.")
            self.real_layer_idxs = None
        return self._real_layers

    #Calculate a job and index number from a given simulation number
    def _get_sim_indices(self,nums):
        init_time = time.time()
        job_nums = np.searchsorted(self._cum_sims,nums,side="right")-1
        sim_nums = nums - self._cum_sims[job_nums]
        sim_nums = [sim_nums[job_nums==x] for x in np.unique(job_nums)]
        job_nums = np.unique(job_nums)
        job_nums = job_nums.astype(int)
        job_nums = np.array([self._jobs[job] for job in job_nums])
        print("time to calculate indices: ", time.time()-init_time)
        return job_nums,sim_nums
    
    #Return the simulation results for specific indices
    def get_results(self,nums):
        pass

    #Randomly sample from the simulation dataset
    def random_sample(self,num_samples=10):
        idxs = np.random.choice(np.arange(0,self.total_sims-1,dtype=int),size=num_samples,replace=False)
        out = self.get_results(idxs)
        return idxs,out

    def save_all_smb_bmb_best_layers(self,
                                    layer_mask=None,
                                    noise = False,
                                    overwrite = False):
        """
        See child classes for docstring
        """
        pass
    

    def check_dsum_iso(self,dsum_iso):
        """
        Checks if the layers fit in the ice shelf and whether they are reasonably continuous

        Parameters

        dsum_iso: np.ndarray containig cumulative thicknesses of layers from the bottom
        """
        thickness =dsum_iso[:,0,-1]
        correct_thickness = np.all(np.abs(thickness- (self.surface-self.base))<tol)
        derivative = np.diff(thickness,axis=0)
        correct_smoothness = np.all(np.abs(derivative)<50)
        layer_number = dsum_iso.shape[-1]
        enough_layers = layer_number>50
        return correct_thickness and correct_smoothness and enough_layers
    
    
    def check_age_iso(self,age_iso):
        """
        Checks if the age of the layers is less than the total simulation time, and whether there exist layers that are sufficiently old to be reasonable guesses for the real layers

        Parameters

        age_iso: np.ndarray containig ages of layers of dsum_iso
        """
        old_enough = np.any(age_iso>100)
        return old_enough
    
    def check_trackers(self,trackers):
        """
        Checks whether the trackers are within the x-domain and that there are enough of them for the simulation to be reasonable

        Parameters

        trackers: np.ndarray containing the xyz-positions of the trackers
        """
        within_domain = np.all(trackers[:,0]>0-tol) and np.all(trackers[:,0]<self.x[-1]+tol)
        enough_trackers = trackers.shape[0]>10
        return within_domain and enough_trackers
    

    def set_anomalies(self, anomalies):
        """
        Set the anomalies for the loader.
        """
        self._anomalies = anomalies


    def set_layer_bounds(self, layer_bounds):
        """
        Set the layer bounds for the loader.
        """
        self._layer_bounds = layer_bounds
        self.masks = [self.x > bound for bound in self._layer_bounds]



    def find_anomalies(self,overwrite = False):
        """
        Load anomalies from file        
        """

        #Check if anomalies have already been identified
        fname = Path(self._fol_path,self._exp_path,"anomalies.p")
        exists = os.path.isfile(fname)
        if exists and not overwrite:
            with open(Path(self._fol_path,self._exp_path,"anomalies.p"), "rb") as handle:
                anomalies = pickle.load(handle)
            self.set_anomalies(anomalies)
            return anomalies
        else:
            raise NotImplementedError("Anomaly detection disabled, you need to save anomalies first.")

    
    def all_layer_xbounds(self,overwrite=False):
        """
        Load layer bounds from file
        """
        fname = Path(self._fol_path,self._exp_path,"layer_bounds" +self.gt_version + ".p")
        exists = os.path.isfile(fname)
        if exists and not overwrite:
            with open(fname, 'rb') as handle:
                layer_bounds = pickle.load(handle)
                self.set_layer_bounds(layer_bounds)
                return layer_bounds
        else:
            raise NotImplementedError("Layer bounds calculation disabled, you need to save layer bounds first.")

        


                            


            


        


