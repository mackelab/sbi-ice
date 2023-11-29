import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern
import logging
import pickle
import numpy as np
import torch
from sbi_ice.simulators import Layer_Tracing_Sim as lts
from sbi_ice.utils import custom_priors,modelling_utils
from sbi.utils import process_prior

#Change config_name for other shelves
@hydra.main(version_base=None, config_path="../../configs/run_sims", config_name="Ekstrom") 
def my_app(cfg : DictConfig)->None:
    modelling_utils.set_seed(cfg.random_seed.seed)
    logging.basicConfig(filename="run_sims.log",encoding="utf-8",level=logging.INFO)
    logging.info("Started Run")
    n_samples = cfg.random_seed.nsims
    geom = lts.Geom(nx_iso=cfg.iso_hparams.nx_iso,ny_iso=cfg.iso_hparams.ny_iso)
    x_setup,bs,ss,vxs,tmb,dQdx,dQdy = lts.init_fields_from_fname(geom,to_absolute_path(cfg.setup_file.address))
    logging.info("Read File")
    xsmb = x_setup.copy()
    smb_prior_length_scale = cfg.prior.length_scale
    smb_prior_nu = cfg.prior.nu

    #Define prior distribution over SMB
    if "GP_mean_mu" in cfg.prior.keys():
        GP_mean_mu = cfg.prior.GP_mean_mu
        GP_mean_sd = cfg.prior.GP_mean_sd
        GP_var_min = cfg.prior.GP_var_min
        GP_var_max = cfg.prior.GP_var_max
        logging.info("GP mean distributed normally")

    elif "GP_mean_min" in cfg.prior.keys():
        GP_mean_min = cfg.prior.GP_mean_min
        GP_mean_max = cfg.prior.GP_mean_max
        GP_var_min = cfg.prior.GP_var_min
        GP_var_max = cfg.prior.GP_var_max
        logging.info("GP mean distributed uniformly")


    logging.info("Set up prior")
    ker = Matern(length_scale=smb_prior_length_scale,nu=smb_prior_nu)
    gpr = GaussianProcessRegressor(kernel=ker)
    mvn_mean,mvn_cov = gpr.predict(xsmb.reshape(-1,1),return_cov=True)
    #ensure covariance matrix is PSD.
    eps = 1e-6
    a = np.zeros(shape = mvn_cov.shape)
    np.fill_diagonal(a,eps)
    mvn_cov += a
    mvn_mean = torch.from_numpy(mvn_mean)
    mvn_cov = torch.from_numpy(mvn_cov)
    if "GP_mean_mu" in cfg.prior.keys():
        custom_prior = custom_priors.AppendedMVN(torch.tensor([GP_mean_mu]),torch.tensor([GP_mean_sd]),torch.tensor([GP_var_min]),
                                             torch.tensor([GP_var_max]),mvn_mean,mvn_cov)
        prior, *_ = process_prior(custom_prior,
                                    custom_prior_wrapper_kwargs=dict(lower_bound=torch.cat((torch.tensor([GP_mean_mu-3*GP_mean_sd,GP_var_min]),-5.0*torch.ones(mvn_mean.size()))), 
                                                                upper_bound=torch.cat((torch.tensor([GP_mean_mu+3*GP_mean_sd,GP_var_max]),5.0*torch.ones(mvn_mean.size())))))

    elif "GP_mean_min" in cfg.prior.keys():
        custom_prior = custom_priors.AppendedMVN2(torch.tensor([GP_mean_min]),torch.tensor([GP_mean_max]),torch.tensor([GP_var_min]),
                                             torch.tensor([GP_var_max]),mvn_mean,mvn_cov)
        prior, *_ = process_prior(custom_prior,
                            custom_prior_wrapper_kwargs=dict(lower_bound=torch.cat((torch.tensor([GP_mean_min,GP_var_min]),-2.0*torch.ones(mvn_mean.size()))), 
                                                            upper_bound=torch.cat((torch.tensor([GP_mean_max,GP_var_max]),2.0*torch.ones(mvn_mean.size())))))

    #Sample from prior
    samples = prior.sample((n_samples,)).numpy()
    logging.info("Sampled from prior")
    bmb_list = []
    smb_cnst_mean_array = samples[:,0]
    smb_sd_array= samples[:,1]
    smb_unperturbed_array = samples[:,2:]
    logging.info(smb_cnst_mean_array.shape)
    logging.info(smb_sd_array.shape)
    logging.info(smb_unperturbed_array.shape)
    logging.info("Divided samples")
    smb_samples = np.expand_dims(smb_cnst_mean_array,1) + np.expand_dims(smb_sd_array,1)*smb_unperturbed_array
    logging.info(smb_sd_array.shape)

    #Simulate for each prior sample.
    dsum_iso_list = []
    age_iso_list = []
    tracker_list = []
    logging.info("Running sims...")
    logging.info(cfg.scheds)
    for j in range(n_samples):
        geom = lts.Geom(nx_iso=cfg.iso_hparams.nx_iso,ny_iso=cfg.iso_hparams.ny_iso)
        x_setup,bs,ss,vxs,tmb,dQdx,dQdy = lts.init_fields_from_fname(geom,to_absolute_path(cfg.setup_file.address))

        smb = smb_samples[j].copy()
        smb_regrid,bmb_regrid = lts.init_mb(geom,x_setup,tmb,smb=smb)
        scheds = cfg.scheds
        init_layers = cfg.iso_hparams.init_layers
        for key in cfg.scheds.keys():
            logging.info(scheds[key])
            sched = lts.Scheduele(**scheds[key])
            geom.initialize_layers(sched,init_layers)
            lts.sim(geom,smb_regrid,bmb_regrid,sched)
        bmb_list.append(bmb_regrid)
        idxs, dsum_iso, age_iso = geom.extract_nonzero_layers()
        active_trackers = geom.extract_active_trackers()
        dsum_iso_list.append(dsum_iso)
        age_iso_list.append(age_iso)
        tracker_list.append(active_trackers)
    logging.info("Sims complete")
    logging.info("Saving results")
    #Save results to pickle file.
    with open("res_batch.p", "wb") as fh:
        pickle.dump(dict(dsum_iso_array=dsum_iso_list,age_iso_array = age_iso_list,smb_unperturbed_array=smb_unperturbed_array,smb_cnst_mean_array = smb_cnst_mean_array,smb_sd_array = smb_sd_array,bmb_array=bmb_list,tracker_array=tracker_list), fh)
    logging.info("Results saved")
if __name__ == "__main__":
    hyperparams = my_app()


