import numpy as np
import torch
from scipy.signal import butter, lfilter, freqz,filtfilt
from sbi_ice.utils.modelling_utils import regrid

rho_ice = 910.0
rho_snow = 450.0
density_decay = 0.033
c_light = 3.0e8


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(x,y, cutoff, fs, order=5):
    nan_mask = np.isnan(y,dtype=bool)
    new_x = x[~nan_mask]
    new_y = y[~nan_mask]


    b, a = butter_lowpass(cutoff, fs, order=order)
    smoothed_y = filtfilt(b, a, new_y)
    return new_x,smoothed_y

def rho(depth):
    """Density-depth relation of ice."""
    if isinstance(depth,np.ndarray):
        return rho_ice -(rho_ice-rho_snow)*np.exp(-density_decay*depth)
    return rho_ice - (rho_ice-rho_snow)*torch.exp(-density_decay*depth)

def ref_ind(z):
    """Refractive index of ice as a function of depth (Looyenga 1965)."""
    return ((1/rho_ice)*rho(z)*(np.cbrt(3.15)-1)+1)**3


def twt2depth(twt,z):
    """Convert two-way travel time to depth, at a resolution set by z."""
    v = c_light/(np.sqrt(ref_ind(z)))

    dts = np.concatenate((np.array([0]),np.diff(z)))/v
    traveltimedepth = np.cumsum(dts)
    depth = np.zeros_like(twt)
    for i in range(0,twt.shape[0]):
        depth[i] = np.nan if not twt[i]>0 else z[np.argmin(np.abs(traveltimedepth-twt[i]/2))]
    return depth

def twt_err(x, freq, PSD_log_mean, PSD_log_var):
    """
    Sample a baseline, x-dependent noise from a given power spectral density.

    Args:
        x (torch.Tensor): Input tensor.
        freq (torch.Tensor): Frequency tensor.
        PSD_log_mean (torch.Tensor): Mean of the log power spectral density.
        PSD_log_var (torch.Tensor): Variance of the log power spectral density.

    Returns:
        torch.Tensor: Noise sample.

    """
    Lx = x.max() - x.min()
    PSD_sample = torch.distributions.MultivariateNormal(loc=PSD_log_mean, covariance_matrix=torch.diag(PSD_log_var)).sample()
    phase_sample = torch.distributions.Uniform(low=torch.zeros_like(PSD_log_mean), high=2*np.pi*torch.ones_like(PSD_log_mean)).sample()

    amplitudes = torch.sqrt(torch.exp(PSD_sample))
    realizations = torch.zeros((PSD_log_mean.size(0), x.size(1)))
    for i in range(PSD_log_mean.size(0)):
        realizations[i, :] = amplitudes[i] * torch.cos(2*torch.pi*freq[i]*x[0].numpy() + phase_sample[i])
    total_realizations = 4e-10 * torch.sum(realizations, axis=0)
    PSD_noise = torch.stack([total_realizations for i in range(x.size(0))])
    return PSD_noise


def depth_correction(z):
    """
    Calculates a depth-dependent scale factor for the noise baseline noise model.

    Args:
        z (torch.Tensor): Input depth values.

    Returns:
        tuple: A tuple containing the correction factor (c_fac) and the cumulative sum of the corrected depth values.
    """

    zd = torch.cat([torch.zeros_like(z[0]).expand(1,z.shape[1]),z],dim=0)
    c_fac =  c_light/(2* torch.sqrt(ref_ind(z)))
    diffs = torch.diff(zd,dim=0)
    return c_fac,torch.cumsum(c_fac*diffs,dim=0)


def picking_error(x,z):
    """Calculate the human error in picking the layers from raw radar data."""
    dt = 1.0e-9
    vels = c_light/(torch.sqrt(ref_ind(z)))
    spatial_res = dt*vels/2
    options = torch.stack([-2*spatial_res, -spatial_res, torch.zeros_like(spatial_res), spatial_res, 2*spatial_res])
    idxs = torch.randint(0,5,x.size())
    picking_err = index_tensor_3d(options, idxs)
    return picking_err

def index_tensor_3d(A, B):
    M, L, N = A.shape
    indices = B.repeat((M,1,1)).reshape((M,L,N))
    result = torch.gather(A, 0, indices)
    return result[0]

def dumb_index_tensor_3d(A, B):
    M, L, N = A.shape
    res= torch.zeros((L,N),dtype=torch.float)
    for i in range(L):
        for j in range(N):
            res[i,j] = A[B[i,j],i,j]
    return res


def depth_error(x, z, freq=None, PSD_log_mean=None, PSD_log_var=None):
    """
    Samples the noise model for all layers in a given simulation.

    Args:
        x (ndarray): x-locations of layer elevations.
        z (ndarray): z-locations of layer elevations
        freq (float, optional): Fourier frequencies of x-discretization
        PSD_log_mean (float, optional): mean of the log power spectral density. Defaults to None.
        PSD_log_var (float, optional): variance of the log power spectral density. Defaults to None.

    Returns:
        tuple: A tuple containing the base error, depth factor, picking error, and the product of base error and depth factor.
    """
    assert x.shape == z.shape, "x and z must have same shape"
    base = twt_err(x, freq, PSD_log_mean, PSD_log_var)
    vels, depth_factor = depth_correction(z)
    picking_err = picking_error(x, z)
    return base, depth_factor, picking_err, base * depth_factor



def best_contour(true_layer,sim_layers,layer_mask = None,method = "MSE"):
    """
    Finds the closes layer (in the mean-square sense) to the GT layer.
    Returns the best layer, along with its layer index and MSE relative to the GT layer.
    """
    if layer_mask is None:
        layer_mask = ~torch.isnan(true_layer)
    trunc_y = sim_layers[:,layer_mask]

    if method == "MSE" or method == "advanced_noise":
        norm = torch.linalg.vector_norm(trunc_y- true_layer[layer_mask].expand(1,trunc_y.shape[1]),ord=2,dim=1)
        temp_norm,idx = torch.min(norm,dim=0)
        sel_y = sim_layers[idx]
    else:
        raise NotImplementedError("method must be one of MSE or advanced_noise")
    return sel_y,temp_norm,idx


def best_contour_diff_xs(x_true,true_layer,x_sim,sim_layers,method="MSE",layer_idx=None):
    """Calculates best_contour when the simulated and GT layers are defined on different grids."""
    try:
        true_layer = regrid(x_true,true_layer,x_sim)
    except:
        true_layer = regrid(x_true,true_layer,x_sim,kind="linear")
    layer_mask = np.ones_like(true_layer,dtype=bool)
    true_layer = torch.tensor(true_layer)
    return best_contour(true_layer,sim_layers,layer_mask,method,layer_idx)

