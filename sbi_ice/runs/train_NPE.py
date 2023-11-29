import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pathlib import Path,PurePath
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern
import logging
import pickle
import torch
from torch.distributions import Normal,Uniform,MultivariateNormal
import numpy as np
from sbi_ice.utils import modelling_utils,custom_priors,posterior_utils,plotting_utils,misc
from sbi_ice.loaders import ShortProfileLoader
from sbi.inference import SNPE_C
import sbi.analysis as analysis
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import process_prior, BoxUniform
from sbi.neural_nets.embedding_nets import CNNEmbedding
import matplotlib.pyplot as plt
from tueplots import figsizes
color_opts = plotting_utils.setup_plots()

#Change config_name for other shelves
@hydra.main(version_base=None, config_path="../../configs/training", config_name="Ekstrom_train")
def my_app(cfg : DictConfig)->None:

    logging.basicConfig(filename="training.log",filemode="w",encoding="utf-8",level=logging.INFO)
    logging.info("Started Training")

    # setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("used device: " + device.type)
    posterior_n = cfg.num_sims_training

    modelling_utils.set_seed(cfg.seed)
    # parse hydra params
    pm = cfg.posterior_model
    training_cfg = cfg.training
    # load data
    paths = cfg.paths
    work_folder = misc.get_project_root()

    #Define dataloader and load prior simulations
    shelf_setup_path = paths.shelf_setup_path
    shelf_folder_path = paths.shelf_folder_path
    exp_path = paths.exp_path
    logging.info("Defining Dataloader...")
    loader = ShortProfileLoader.ShortProfileLoader(Path(work_folder,shelf_setup_path),Path(work_folder,shelf_folder_path),exp_path,gt_version = cfg.gt_version)

    logging.info("Data loader object created")

    contour_arrays,norm_arrays,age_arrays,smb_unperturbed_all,smb_cnst_means_all,smb_sds_all,smb_all,bmb_all = loader.load_training_data(layers_fname = cfg.layers_fname,mb_fname = cfg.mb_fname)
    logging.info("Data loaded")


    #Downsample the arrays for smaller x and theta dimensionalities
    smb_restored = (smb_all - torch.unsqueeze(smb_cnst_means_all,1))/torch.unsqueeze(smb_sds_all,1)

    layer_mask = torch.from_numpy(loader.masks[cfg.grid.layer_idx])
    layer_sparsity = cfg.grid.layer_sparsity
    smb_sparsity = cfg.grid.smb_sparsity

    layers = contour_arrays[cfg.grid.layer_idx]
    layer_mask_slice = torch.zeros(layers.shape[-1], dtype=bool)
    layer_mask_slice[::layer_sparsity] = 1

    layer_all_mask = layer_mask * layer_mask_slice
    layers_all = layers[:,layer_all_mask].to(device).float()
    logging.info("layers_all shape:")
    logging.info(layers_all[0].shape)
    smb_mask_slice = torch.zeros(layers.shape[-1],dtype=bool)
    smb_mask_slice[::smb_sparsity] = 1
    #smb_mask = layer_mask*smb_mask_slice
    smb_mask = smb_mask_slice #infer all smb values
    smb = smb_restored[:,smb_mask]
    theta = smb_all[:,smb_mask].to(device).float()
    smb_x = loader.x[smb_mask]

    logging.info("Converted Output to theta and x")

    #Define prior over SMB
    GP_mean_mu = torch.tensor([loader.cfg.prior.GP_mean_mu],device=device)
    GP_mean_sd = torch.tensor([loader.cfg.prior.GP_mean_sd],device=device)
    GP_var_min = torch.tensor([loader.cfg.prior.GP_var_min],device=device)
    GP_var_max = torch.tensor([loader.cfg.prior.GP_var_max],device=device)
    smb_prior_length_scale = loader.cfg.prior.length_scale
    smb_prior_nu = loader.cfg.prior.nu

    #Define GP kernel
    ker = Matern(length_scale=smb_prior_length_scale,nu=smb_prior_nu)
    gpr = GaussianProcessRegressor(kernel=ker)

    mvn_mean,mvn_cov = gpr.predict(smb_x.reshape(-1,1),return_cov=True)
    eps = 1e-6
    a = np.zeros(shape = mvn_cov.shape)
    np.fill_diagonal(a,eps)
    mvn_cov += a
    mvn_mean = torch.from_numpy(mvn_mean).to(device)
    logging.info(GP_mean_mu)
    logging.info(GP_mean_sd)
    logging.info(GP_var_min)
    logging.info(GP_var_min)
    logging.info(mvn_mean)
    mvn_cov = torch.from_numpy(mvn_cov).to(device)


    #custom_prior = custom_priors.AppendedMVN(GP_mean_mu,GP_mean_sd,GP_var_min,GP_var_max,mvn_mean,mvn_cov)
    #custom_prior = torch.distributions.Uniform(low = -5.0*torch.ones(mvn_mean.size()),high = 5.0*torch.ones(mvn_mean.size()))
    #In single round NPE the prior is defined entirely from the simulations so we can use this hack for the prior here.
    custom_prior = BoxUniform(low=-5.0*torch.ones(mvn_mean.size()), 
                                high=5.0*torch.ones(mvn_mean.size()),device=device.type)
    sample = custom_prior.sample(torch.Size((1,)))

    logging.info(sample)
    logging.info("Defined prior")
    # Keeping only the first return.
    # prior, *_ = process_prior(custom_prior,
    #                           custom_prior_wrapper_kwargs=dict(lower_bound=torch.cat((torch.tensor([GP_mean_mu-3*GP_mean_sd,GP_var_min]),-5.0*torch.ones(mvn_mean.size()))), 
    #                                                            upper_bound=torch.cat((torch.tensor([GP_mean_mu+3*GP_mean_sd,GP_var_max]),5.0*torch.ones(mvn_mean.size())))))
    prior, *_ = process_prior(custom_prior,
                              custom_prior_wrapper_kwargs=dict(lower_bound=-5.0*torch.ones(mvn_mean.size()), 
                                                               upper_bound=5.0*torch.ones(mvn_mean.size())))

    logging.info((pm.model,pm.hidden_features,pm.num_components,pm.num_transforms,pm.num_bins))
    logging.info(pm.embedding_net.include)
    #Define embedding net for NPE
    if pm.embedding_net.include:
        logging.info("Embedding net included")

        if pm.embedding_net.type == "FCN":
            logging.info("Fully Connected Embedding")

            emb_net = posterior_utils.FCEmbedding(input_dim=layers_all[0].shape[-1],
                                                output_dim=pm.embedding_net.output_dim,
                                                num_layers=pm.embedding_net.num_layers,
                                                num_hiddens=pm.embedding_net.num_hiddens)
        elif pm.embedding_net.type == "CNN":
            logging.info("Convolutional Embedding")
            logging.info(pm.embedding_net.out_channels_per_layer)
            emb_net = CNNEmbedding(input_shape = layers_all[0].shape,
                                in_channels = pm.embedding_net.in_channels,
                                out_channels_per_layer = pm.embedding_net.out_channels_per_layer,
                                num_conv_layers = pm.embedding_net.num_conv_layers,
                                num_linear_layers = pm.embedding_net.num_linear_layers,
                                num_linear_units = pm.embedding_net.num_linear_units,
                                output_dim = pm.embedding_net.output_dim,
                                kernel_size = pm.embedding_net.kernel_size,
                                pool_kernel_size = pm.embedding_net.pool_kernel_size)
            logging.info("Convolutional Embedding defined")

        else:
            logging.WARN("No such embedding net type defined! Using identity...")
            emb_net = torch.nn.Identity()
    else:
        logging.info("Embedding net  not included")
        emb_net = torch.nn.Identity()
    #Define Density Estimator.
    density_estimator_build_fun = posterior_nn(
        model=pm.model,
        hidden_features=pm.hidden_features,
        num_components=pm.num_components,
        num_transforms = pm.num_transforms,
        num_bins = pm.num_bins,
        z_score_x=None,  # None
        z_score_theta=None,  #'independent'
        embedding_net=emb_net,  #embedding_net
    )
    logging.info("Defined Density Estimator")
    #Define inference object
    inference = SNPE_C(
        prior=prior,
        density_estimator=density_estimator_build_fun,
        device = device.type,
        logging_level = "DEBUG",
        show_progress_bars=False
    )
    logging.info("Defined inference")
    logging.info((theta.shape,layers_all.shape))
    inference = inference.append_simulations(theta[:posterior_n], layers_all[:posterior_n])
    logging.info("Added simulations to inference object")
    logging.info("Training Density Estimator...")
    #Train on data.
    inference.train(
        max_num_epochs=10_000, 
        training_batch_size=training_cfg.batchsize,
        learning_rate=5e-3,  # 5e-4,
        force_first_round_loss=True,
        validation_fraction=0.1,  # fraction of data used for validation. Default=0.1
        stop_after_epochs=20,  # number of epochs to wait before training is stopped. Default=20
    )
    logging.info("Training Done")
    logging.info(inference.summary)
    #Save inference results.
    with open("inference.p", "wb") as f:
        pickle.dump(dict(inference=inference,prior=prior,smb_mask = smb_mask,layer_mask=layer_all_mask), f)
    logging.info("Model saved")


    logging.info("Plotting Posterior Pairplot")
    #Condition on GT layer and plot.
    true_layer = torch.tensor(loader.real_layers[cfg.grid.layer_idx][layer_all_mask]).float()
    logging.info(true_layer)
    cpu_nn = inference._neural_net.to("cpu")
    
    cpu_custom_prior = BoxUniform(low=-5.0*torch.ones(mvn_mean.size()), 
                                high=5.0*torch.ones(mvn_mean.size()))
    cpu_prior, *_ = process_prior(cpu_custom_prior,
                              custom_prior_wrapper_kwargs=dict(lower_bound=-5.0*torch.ones(mvn_mean.size()), 
                                                               upper_bound=5.0*torch.ones(mvn_mean.size())))

    posterior = inference.build_posterior(cpu_nn,cpu_prior)
    posterior.set_default_x(true_layer)
    samples = posterior.sample((1000,))

    #Pairplot
    plt.rcParams.update(figsizes.icml2022_full())
    subset = [0,1]
    limits = []
    for i in range(0,samples.shape[-1]):
        limits.append([-2,2])
    ctr = 0
    while ctr<samples.shape[-1]:
        subset.append(ctr)
        ctr+=5
    logging.info("Plotting...")
    try:
        true_smb_unperturbed = modelling_utils.regrid(loader._true_xmb,loader._true_smb_unperturbed,loader.x)
        true_smb_unperturbed_ds = true_smb_unperturbed[smb_mask]
        gt_param = loader._true_smb_const_mean + loader._true_smb_var*true_smb_unperturbed_ds
        points = gt_param

    except:
        points = []
    fig,axs = analysis.pairplot(samples,subset= subset,limits=limits,labels=[],points=points,points_offdiag={
                "marker": ".",
                "markersize": 5,
            },points_colors=["darkorange"])

    fig.savefig("posterior_pairplot.png")
    logging.info("Plotting Posterior Spatially")

    #Spatial plot of posterior
    spatial_samples = samples
    try:
        true_smb = true_smb_unperturbed*loader._true_smb_var+loader._true_smb_const_mean
    except:
        true_smb = None
    perm = torch.randperm(smb_all.size(0))
    idx = perm[:1000]
    prior_spatial_samples = smb_all[idx][:,smb_mask]
    tmb = smb_all[0].numpy() + bmb_all[0].numpy()
    fig,axs = plotting_utils.plot_posterior_spatial(loader.x,
                                            smb_mask,
                                            tmb,
                                            prior_spatial_samples.to("cpu"),
                                            spatial_samples.to("cpu"),
                                            true_smb=true_smb,
                                            plot_samples=False
                                            )
    fig.savefig("posterior_spatial.png")
    logging.info("Returning Best Validation logprob")

    best_val_log_prob = inference.summary["best_validation_log_prob"][0]
    return -best_val_log_prob

if __name__ == "__main__":
    hyperparams = my_app()


