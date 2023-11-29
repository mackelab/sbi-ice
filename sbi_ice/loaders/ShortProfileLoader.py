from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import pandas as pd
import os
import pickle
import torch
from sbi_ice.utils.modelling_utils import regrid,regrid_all
from sbi_ice.utils.noise_model import best_contour,depth_error
from sbi_ice.loaders.BaseLoader import BaseLoader,tol
import logging

logger = logging.getLogger(__name__)


class ShortProfileLoader(BaseLoader):
    """
    This class handles all data for a set of simulations (e.g. Ekstrom/exp2).
    This includes the inpute files to the simulations, the raw simulation outputs, the processed simulation
    outputs, and the ground truth values.
    This extends BaseLoader for the way we define SMB, BMB and the best layers.

    Args:
        setup_dir (str): The directory containing the setup files (input to simulation).
        fol_path (str): The path to the folder containing the experiment files (which shelf?).
        exp_path (str): The path to the experiment files (which experiment?).
        sims_path (str, optional): The path to the layer simulation files. One of "layer_sims" or "calib_sims".
        gt_version (str, optional): The version of the ground truth mass balance and real layer elevations. Defaults to "v0".
    """
    
    def __init__(self, setup_dir, fol_path, exp_path, sims_path="layer_sims", gt_version="v0"):
        super().__init__(setup_dir, fol_path, exp_path, sims_path, gt_version)
        try:
            self.set_real_mb()
        except:
            print("No `real_mb.p` file detected. If this is a synthetic experiment, save the ground truth mass balance to compare to later.")

    def convert_smb(self, offset, scale, smb):
        """
        Converts the SMB values using an offset and scale.

        Args:
            offset (float): The offset value.
            scale (float): The scale value.
            smb (numpy.ndarray): The SMB values.

        Returns:
            numpy.ndarray: The converted SMB values.
        """
        temp = offset + scale * smb
        return regrid(self.x_coords, temp, self.x)

    def convert_all_smb(self, offsets, scales, smb):
        """
        Converts all SMB values using offsets and scales.

        Args:
            offsets (numpy.ndarray): The offset values.
            scales (numpy.ndarray): The scale values.
            smb (numpy.ndarray): The SMB values.

        Returns:
            numpy.ndarray: The converted SMB values.
        """
        temp = offsets + scales * smb
        return regrid_all(self.x_coords, temp, self.x)

    def set_real_mb(self):
        """
        Set the real mass balance values by loading them from a file.

        Returns:
            int: 0 indicating successful execution.
        """
        with open(self._real_mb_file, "rb") as fh:
            out = pickle.load(fh)
        self._true_xmb = out['x_coord']
        self._true_smb_const_mean = out['smb_const_mean']
        self._true_smb_var = out["smb_var"]
        self._true_smb_unperturbed = out["smb_unperturbed"]
        self._true_bmb_regrid = out["bmb_regrid"]
        return 0

    def get_results(self, nums):
        """
        Retrieves results for the given job numbers and simulation numbers.

        Args:
            nums (list): List of numbers between 0 and self.total_sims.

        Returns:
            dict: A dictionary containing the retrieved results.
                - 'dsum_isos': List of dsum_iso values for all requested sims.
                - 'age_isos': List of age_iso values for all requested sims.
                - 'smb_unperturbed': NumPy array of smb_unperturbed values for all requested sims.
                - 'smb_cnst_mean': NumPy array of smb_cnst_mean values for all requested sims.
                - 'smb_sd': NumPy array of smb_sd values for all requested sims.
                - 'bmb_array': NumPy array of bmb_array values for all requested sims.
                - 'trackers': List of tracker values for all requested sims.
        """
        job_nums, sim_nums = self._get_sim_indices(nums)
        out = {}
        dsum_iso_list = []
        age_iso_list = []
        smb_unperturbed_list = []
        smb_cnst_mean_list = []
        smb_sd_list = []
        bmb_regrid_list = []
        tracker_list = []

        for i in range(0, job_nums.size):
            with open(Path(self._layer_sims_path, str(job_nums[i]), 'res_batch.p'), 'rb') as handle:
                res = pickle.load(handle)
            for j in sim_nums[i]:
                dsum_iso_list.append(res['dsum_iso_array'][j])
                age_iso_list.append(res['age_iso_array'][j])
                smb_unperturbed_list.append(res['smb_unperturbed_array'][j])
                smb_cnst_mean_list.append(res['smb_cnst_mean_array'][j])
                smb_sd_list.append(res['smb_sd_array'][j])
                bmb_regrid_list.append(res['bmb_array'][j])
                tracker_list.append(res['tracker_array'][j])

        out['dsum_isos'] = dsum_iso_list
        out['age_isos'] = age_iso_list
        out['smb_unperturbed'] = np.array(smb_unperturbed_list)
        out['smb_cnst_mean'] = np.array(smb_cnst_mean_list)
        out['smb_sd'] = np.array(smb_sd_list)
        out['bmb_array'] = np.array(bmb_regrid_list)
        out['trackers'] = tracker_list
        return out

    def load_training_data(self, layers_fname, mb_fname):
        """
        Loads training data from saved files.

        Parameters:
            layers_fname (str): The filename of the layers file.
            mb_fname (str): The filename of the mass balance file.

        Returns:
            tuple: A tuple containing the following arrays:
                - contour_arrays: Noisy best layers for each simulation.
                - norm_arrays: Distance between best layers and GT layer.
                - age_arrays: Ages of best layers.
                - smb_unperturbed_all: SMB array values before convert_smb for all simulations.
                - smb_cnst_means_all: SMB constant values to shift by for all simulations
                - smb_sds_all: SMB standard deviations to scale by for all simulations.
                - smb_all: Converted SMB arrays used for all simulations
                - bmb_all: BMB arrays used for all simulations.

        Raises:
            FileNotFoundError: If no saved files are found. Create them first using select_layers.py.
        """
        self.all_layer_xbounds(overwrite=False)
        anomalies = self.find_anomalies()
        layers_fname = Path(self._layer_sims_path, "..", layers_fname)
        mb_fname = Path(self._layer_sims_path, "..", mb_fname)
        logger.info(layers_fname)
        logger.info(mb_fname)
        saved = os.path.exists(layers_fname) and os.path.exists(mb_fname)
        if saved:
            print('Files already exist! Loading results...')

            with open(layers_fname, "rb") as fh:
                contour_arrays, norm_arrays, age_arrays = pickle.load(fh).values()
            with open(mb_fname, "rb") as fh:
                smb_unperturbed_all, smb_cnst_means_all, smb_sds_all, smb_all, bmb_all = pickle.load(fh).values()
        elif not saved:
            if not os.path.exists(layers_fname):
                print('No layers file found!')
            elif not os.path.exists(mb_fname):
                print('No mass balance file found!')

            raise FileNotFoundError("No saved files found. Create them first using select_layers.py")

        return contour_arrays, norm_arrays, age_arrays, smb_unperturbed_all, smb_cnst_means_all, smb_sds_all, smb_all, bmb_all
