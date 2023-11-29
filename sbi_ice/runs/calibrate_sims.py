from pathlib import Path,PurePath
from hydra.utils import to_absolute_path
import numpy as np
import torch
from sbi_ice.loaders import BaseLoader,ShortProfileLoader
from sbi_ice.utils.modelling_utils import regrid,regrid_all
from sbi_ice.utils.noise_model import best_contour
from scipy.signal import butter, lfilter, freqz,filtfilt,periodogram
import joblib
import pickle
import logging
import multiprocessing



# shelf = "Synthetic_long"
# exp = "exp3"
shelf = "Ekstrom"
exp = "exp2"
hydra_dir = to_absolute_path("../../out/" + shelf)
setup_dir = to_absolute_path("../../data/" + shelf + "/setup_files")
gt_version = "v0"
selection_method = "advanced_noise"
anomalies_overwrite = True
layer_bounds_overwrite = False
noise_dist_overwrite = False
sim_or_calib = "layer_sims"

logging.basicConfig(filename=Path("../../out/",shelf,exp,"calib_log.log"),filemode="w",encoding="utf-8",level=logging.INFO)
nprocess = 1
# nprocess = multiprocessing.cpu_count()
logging.info("num_processes: " + str(nprocess))

loader = ShortProfileLoader.ShortProfileLoader(Path(setup_dir),Path(hydra_dir),exp,sims_path =sim_or_calib,gt_version=gt_version)
logging.info("defined loader")

fs = 1/np.diff(loader.x).mean() 


def check_anomaly(res,sim):
    logging.info("Checking anomaly for sim: " + str(sim))


    dsum_isos = res["dsum_iso_array"]
    age_isos = res["age_iso_array"]
    trackers_arr = res["tracker_array"]
    dsum_iso = dsum_isos[sim]
    age_iso = age_isos[sim]
    trackers = trackers_arr[sim]

    check_dsum_iso = loader.check_dsum_iso(dsum_iso)
    check_age_iso = loader.check_age_iso(age_iso)
    check_trackers = loader.check_trackers(trackers)
    #If any of the checks failed, flag anomaly and save which tests it failed
    return check_dsum_iso,check_age_iso,check_trackers

def layer_bounds(res,job, sim):
    logging.info("Calculating layer bounds for sim: " + str(sim))
    bounds = []
    if (job,sim) not in loader._anomalies:
        dsum_iso = res["dsum_iso_array"][sim]
        trackers = res["tracker_array"][sim]

        #Take z-coordinates of trackers of unique x-coordinates
        unique_trackers = np.unique(trackers,axis=0)
        #Sort the trackers by x-coordinate
        sorted_trackers = unique_trackers[unique_trackers[:,0].argsort()]
        xpoints = sorted_trackers[:,0]
        zpoints = sorted_trackers[:,2]
        for layer in loader.real_layers:
            logging.info(layer.shape)
            nan_mask = np.isnan(layer,dtype=bool).copy()
            logging.info(nan_mask.shape)
            layer_x = loader.x[~nan_mask].copy()
            #Regrid the real layer data onto the tracker x-coordinates
            regrid_layer = regrid(layer_x,layer[~nan_mask],xpoints)
            diff = zpoints-regrid_layer
            #Find first location where the trackers go from being above the layer to below the layer (diff goes from positive to negative)
            signchange = np.argwhere(np.diff(np.sign(diff))<0)
            if np.all(diff<0):
                bound = [layer_x[0]]
            else:
                try:
                    bound = xpoints[signchange[0]]
                except: 
                    bound = [loader.x[-1]]
            bounds.append(bound[0])
        logging.info(bounds)
        return np.array(bounds)
    else:
        return None
    
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(layer, cutoff, fs, order=5):
    nan_mask = np.isnan(layer,dtype=bool).copy()
    layer = layer[~nan_mask]


    b, a = butter_lowpass(cutoff, fs, order=order)
    # y = lfilter(b, a, layer)
    y = filtfilt(b, a, layer)
    return nan_mask,y
def calculate_normalized_PSD(layer):
    nan_mask = np.isnan(layer,dtype=bool).copy()
    layer = layer[~nan_mask]
    layer = layer - layer.mean()
    (freq, PSD) = periodogram(layer, fs, detrend = "constant",scaling='spectrum')
    total_PSD = PSD.sum()
    return (freq,PSD,total_PSD)

def calculate_noise_coeffs(res,job,sim):
    """Generate noise with the same power spectral density as differences between simulated and real layers."""
    logging.info("Calculating noise coeffs for sim: " + str(sim))
    cutoff = cutoff = 1.0/25.0e3

    # Calculate the PSD distribution
    freqs = []
    PSD_diffs = []
    total_PSD_diffs = []
    #Detrend sims and calculate PSD
    smoothed_layers = []
    loader.all_layer_xbounds()
    logging.info("smooothing real layers")
    for i in range(loader.real_layers.shape[0]):
        layer_x, smooth_layer = butter_lowpass_filter(loader.real_layers[i], cutoff, fs, order=5)
        smoothed_layers.append(smooth_layer)

    if (job,sim) not in loader._anomalies:
        dsum_iso = res["dsum_iso_array"][sim]
        trackers = res["tracker_array"][sim]
        logging.info("Calculating PSD")
        logging.info(len(loader.masks))
        for layer_index,layer in enumerate(loader.real_layers):
            best_layer,temp_norm,idx = best_contour(torch.Tensor(layer),torch.Tensor(dsum_iso[:,0,:].T+loader.base),layer_mask = loader.masks[layer_index],method = "MSE")
            logging.info("Smoothing best layer")
            nan_mask,smoothed_sim = butter_lowpass_filter(best_layer.numpy(),cutoff,fs,order=5)
            logging.info("Calculating residuals in real")
            noise_real = loader.real_layers[layer_index,~np.isnan(loader.real_layers[layer_index],dtype=bool)] - smoothed_layers[layer_index]
            logging.info("Calculating residuals in sim")
            noise_sim = (dsum_iso[:,0,idx]+loader.base.flatten())[~np.isnan(loader.real_layers[layer_index],dtype=bool)].T - smoothed_sim[~np.isnan(loader.real_layers[layer_index],dtype=bool)]
            logging.info("Calculating PSD for layer")
            (freq, PSD_temp,PSD_tot_temp) = calculate_normalized_PSD(noise_sim-noise_real)
            freq = freq[1:]
            freqs.append(freq)
            PSD_diffs.append(PSD_temp[1:])
        logging.info("Finished calculating PSD")
        PSD_log_diffs = [np.log(diffs) for diffs in PSD_diffs]

        return freqs,PSD_log_diffs
    else:
        return None


with joblib.Parallel(n_jobs=nprocess,backend="multiprocessing") as parallel:
    logging.info("In parallel jobs")
    if anomalies_overwrite == False:
        try:
            with open(Path(loader._fol_path,loader._exp_path,"anomalies" + loader.gt_version +".p"), 'rb') as handle:
                anomalies = pickle.load(handle)
                loader.set_anomalies(anomalies)
        except:
            pass
    else:
        logging.info("Calculating anomalies")
        anomalies = {}
        for job_idx,job in enumerate(loader._jobs):
            sims = np.arange(loader._num_sims[job_idx],dtype=int)
            with open( Path(loader._layer_sims_path,str(job),'res_batch.p'), 'rb') as handle:
                out = pickle.load(handle)
            logging.info("loaded results")
            res = parallel(joblib.delayed(check_anomaly)(out,sim) for sim in sims)
            for sim in sims:
                if (not res[sim][0]) or not (res[sim][1]) or (not res[sim][2]):
                    anomalies[(job,sim)] = {"dsum_iso":res[sim][0],"age_iso":res[sim][1],"trackers":res[sim][2]}
            loader.set_anomalies(anomalies)
        with open(Path(loader._fol_path,loader._exp_path,"anomalies"  +loader.gt_version +".p"), 'wb') as handle:
            pickle.dump(anomalies, handle)
            logging.info("Saved anomalies")
            logging.info(anomalies)
    logging.info("Anomalies")
    logging.info(loader._anomalies)
    if layer_bounds_overwrite == False:
        try:
            with  open(Path(loader._fol_path,loader._exp_path,"layer_bounds" +loader.gt_version + ".p"),"rb") as handle:
                layer_bounds = pickle.load(handle)
            loader.set_layer_bounds(layer_bounds)

        except:
            pass

    else:
        logging.info("Calculating layer bounds")
        bounds = []
        for job_idx,job in enumerate(loader._jobs):
            sims = np.arange(loader._num_sims[job_idx],dtype=int)
            with open( Path(loader._layer_sims_path,str(job),'res_batch.p'), 'rb') as handle:
                out = pickle.load(handle)
            logging.info("loaded results")
            res = parallel(joblib.delayed(layer_bounds)(out,job,sim) for sim in sims)
            for sim in sims:
                if res[sim] is not None:
                    bounds.append(res[sim])
            logging.info(bounds)
            bounds = np.vstack(bounds)
            layer_bounds = np.percentile(bounds,75,axis=0)
            loader.set_layer_bounds(layer_bounds)

        with open(Path(loader._fol_path,loader._exp_path,"layer_bounds" +loader.gt_version + ".p"), 'wb') as handle:
            pickle.dump(layer_bounds, handle)
        logging.info("Saved layer bounds")
        logging.info(layer_bounds)
    logging.info("Layer Masks")
    logging.info(loader._layer_bounds)
    logging.info(len(loader.masks))

    if noise_dist_overwrite == False or shelf != "Ekstrom":
        try:
            with open(Path(loader._fol_path,loader._exp_path,"PSD_matched_noise.p"), 'rb') as handle:
                noise_dist = pickle.load(handle)
        except:
            pass

    else:
        logging.info("Calculating noise distribution")
        noise_dist = {}
        for job_idx,job in enumerate(loader._jobs):
            sims = np.arange(loader._num_sims[job_idx],dtype=int)
            # sims = np.arange(100,dtype=int)
            with open( Path(loader._layer_sims_path,str(job),'res_batch.p'), 'rb') as handle:
                out = pickle.load(handle)
            logging.info("loaded results")
            res = parallel(joblib.delayed(calculate_noise_coeffs)(out,job,sim) for sim in sims)
            logging.info("Computed all PSDs")
            logging.info("Removing None values")
            res = [res[i] for i in range(len(res)) if res[i] is not None]
            freqs = res[0][0]
            freqs = [torch.Tensor(freqs[i]) for i in range(len(freqs))]
            logging.info("freqs:")
            logging.info(freqs)
            PSD_log_diffs = [res[i][1] for i in range(len(res))]
            PSD_log_diffs_means = []
            PSD_log_diffs_vars = []
            for j in range(len(PSD_log_diffs[0])):
                log_diffs = torch.Tensor([PSD_log_diffs[i][j] for i in range(len(PSD_log_diffs))])
                PSD_log_diffs_means.append(torch.mean(log_diffs,dim=0))
                PSD_log_diffs_vars.append(torch.var(log_diffs,dim=0))
            logging.info("PSD_log_diffs_means:")
            logging.info(PSD_log_diffs_means)
            logging.info("PSD_log_diffs_means shape: " + str(PSD_log_diffs_means[0].shape))
            PSD_dict = {"freqs":freqs,"PSD_log_diffs_means":PSD_log_diffs_means,"PSD_log_diffs_vars":PSD_log_diffs_vars}

        with open(Path(loader._fol_path,loader._exp_path,"PSD_matched_noise.p"), 'wb') as handle:
            pickle.dump(PSD_dict, handle)
        logging.info("Saved noise distribution")
        logging.info(PSD_dict)
