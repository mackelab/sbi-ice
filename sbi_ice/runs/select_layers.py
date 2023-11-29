from pathlib import Path,PurePath
from hydra.utils import to_absolute_path
import numpy as np
import torch
from sbi_ice.loaders import BaseLoader,ShortProfileLoader
from sbi_ice.utils.modelling_utils import regrid
from sbi_ice.utils.noise_model import best_contour,depth_error
import sys
import pickle
import logging


def save_all_smb_bmb_best_layers(job,loader,PSD_dict,selection_method="MSE",noise=True):
    """
    Calculates the best fitting layer (to the gt) for each simulation. Returns the contour, norm, and age of the best fitting layer for each simulation
    along with the corresponding smb and bmb for each simulation. Saves the results in a pickle file or returns existing saved file.

    Args:
    selection_method (str, optional): Method for selecting the best layer for each simulation. Defaults to "MSE".
    noisy (bool, optional): Whether or not to add noise to the layers.
    overwrite (bool, optional): Whether or not to overwrite existing saved file. Defaults to False.
    """
    logger = logging.getLogger(__name__)
    if type(job) == int:
        job_num = loader._jobs.index(job)
        job = str(job)

    if type(job) == str:
        job_num = loader._jobs.index(int(job))

    logger.info("Starting job " + job)


    freq = PSD_dict["freqs"][0]
    PSD_log_mean = PSD_dict["PSD_log_diffs_means"][0]
    PSD_log_var = PSD_dict["PSD_log_diffs_vars"][0]
    logger.info("loaded PSD")


    with open( Path(loader._layer_sims_path,job,'res_batch.p'), 'rb') as handle:
        res = pickle.load(handle)
    smbs_unperturbeds = np.array(res['smb_unperturbed_array'],dtype=np.float64)
    smb_unperturbeds_all = smbs_unperturbeds
    smb_cnst_means = np.array(res['smb_cnst_mean_array'],dtype=np.float64)
    smb_sds = np.array(res['smb_sd_array'],dtype=np.float64)
    bmbs = torch.tensor(np.array(res['bmb_array']))[:,:,0]

    dsum_isos = res['dsum_iso_array']
    age_isos = res['age_iso_array']
    logger.info("loaded res")

    non_anomalies = []
    smb_all = []
    contour_arrays = np.zeros((loader.real_layers.shape[0],loader._num_sims[job_num],loader.x.shape[0]))
    norm_arrays = np.zeros((loader.real_layers.shape[0],loader._num_sims[job_num]))
    age_arrays = np.zeros((loader.real_layers.shape[0],loader._num_sims[job_num]))
    for sim in range(0,loader._num_sims[job_num]):
    #for sim in range(0,100):
        if sim % 100 == 0:
            logger.info("job: " + job + "    sim: " + str(sim))
        if (job,sim) not in loader._anomalies:
            non_anomalies.append(sim)
            smb_all.append(torch.from_numpy(loader.convert_smb(smb_cnst_means[sim],smb_sds[sim],smbs_unperturbeds[sim])))
            
            dsum_iso = dsum_isos[sim]
            age_iso = age_isos[sim]
            layers = torch.from_numpy(dsum_iso[:,0,:]).T
            if noise:
                layers_thickness = dsum_iso[:,0,:].T
                heights = layers_thickness + loader.base.flatten()
                layer_depths = loader.surface.flatten() - heights
                layer_depths = torch.from_numpy(layer_depths)
                layer_depths = torch.flip(layer_depths,dims=(0,))
                layer_xs = torch.stack([torch.tensor(loader.x) for i in range(layer_depths.shape[0])])
                if selection_method == "advanced_noise":
                    base_error,depth_corr,picking_error,error = depth_error(layer_xs,layer_depths,freq,PSD_log_mean,PSD_log_var)

                else:
                    # logger.info("Using " + selection_method + " method")
                    base_error,depth_corr,picking_error,error = depth_error(layer_xs,layer_depths)
                flipped_error = torch.flip(error,dims=(0,))
                layers = layers + flipped_error
            for lidx in range(0,loader.real_layers.shape[0]):
                layer_idx = loader.real_layer_idxs[lidx] if loader.real_layer_idxs is not None else None
                true_layer = torch.from_numpy(loader.real_layers[lidx])
                contour,norm,aidx = best_contour(true_layer,layers+loader.base.T,layer_mask=loader.masks[lidx],method = "MSE")
                # print("Calculated best layers")
                contour_arrays[lidx,sim,:] = contour
                norm_arrays[lidx,sim] = norm
                age_arrays[lidx,sim] = age_iso[aidx]




    logger.info("Combining results")
    contour_arrays = torch.from_numpy(contour_arrays[:,non_anomalies,:])
    norm_arrays = torch.from_numpy(norm_arrays[:,non_anomalies])
    age_arrays = torch.from_numpy(age_arrays[:,non_anomalies])
    smb_unperturbed_all = torch.from_numpy(smb_unperturbeds_all[non_anomalies])
    smb_cnst_means_all = torch.from_numpy(smb_cnst_means[non_anomalies])
    smb_sds_all = torch.from_numpy(smb_sds[non_anomalies])
    smb_all = torch.stack(smb_all)
    bmb_all = bmbs[non_anomalies]

    return contour_arrays,norm_arrays,age_arrays,smb_unperturbed_all,smb_cnst_means_all,smb_sds_all,smb_all,bmb_all


if __name__ == "__main__":
    shelf,exp,gt_version,selection_method,job = sys.argv[1:]
    hydra_dir = to_absolute_path("../../out/" + shelf)
    setup_dir = to_absolute_path("../../data/" + shelf + "/setup_files")
    logging.basicConfig(filename=Path("../../out",shelf,exp,"layer_sims",job,"process_layers.log"),filemode="w",encoding="utf-8",level=logging.INFO)
    loader = ShortProfileLoader.ShortProfileLoader(Path(setup_dir),Path(hydra_dir),exp,sims_path ="layer_sims",gt_version=gt_version)
    logger = logging.getLogger(__name__)

    logger.info("defined loader")
    anomalies = loader.find_anomalies()
    logger.info("found anomalies")
    logger.info(anomalies)
    loader.all_layer_xbounds()
    logger.info("defined xbounds")
    selection_method_names = {"MSE":"MSE","advanced_noise":"advanced_noise"}

    processed_fname = Path(loader._fol_path, loader._exp_path,"layer_sims",job, "all_layers.p")
    mb_fname = Path(loader._fol_path, loader._exp_path,"layer_sims",job,"all_mbs.p")

    PSD_dict = pickle.load(Path(loader._fol_path,loader._exp_path,"PSD_matched_noise" +loader.gt_version + ".p").open("rb"))



    res = save_all_smb_bmb_best_layers(job,loader,PSD_dict,selection_method,True)
    logger.info("Finished calculating noise")
    contour_arrays = res[0]
    norm_arrays = res[1]
    age_arrays = res[2]
    smb_unperturbed_all = res[3]
    smb_cnst_means_all = res[4]
    smb_sds_all = res[5]
    smb_all = res[6]
    bmb_all = res[7]

    logger.info("Saving...")
    with open(processed_fname, "wb") as fh:
        pickle.dump({"contour_arrays":contour_arrays,"norm_arrays":norm_arrays,"age_arrays":age_arrays}, fh)
    with open(mb_fname, "wb") as fh:
        pickle.dump({"smb_unperturbed_all":smb_unperturbed_all,"smb_cnst_means_all":smb_cnst_means_all,"smb_sds_all":smb_sds_all,"smb_all":smb_all,"bmb_all":bmb_all}, fh)
    logger.info("Done!")

