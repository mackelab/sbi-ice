import numpy as np
import torch
import random
from scipy.interpolate import interp1d,UnivariateSpline


def set_seed(seed: int):
    """Set the random seed for all backends."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def trunc(values, decs=0):
    """Truncates a numpy array to a certain number of decimals."""
    return np.trunc(values*10**decs)/(10**decs)

def regrid(x,y,xnew,kind="cubic"):
    """Interpolate a function between different discretizations of the same spatial domain"""
    if y is None:
        return None
    #TODO: Currently allowing extrapolation, but this should throw a warning if we are extrapolating too far
    f =  interp1d(x,y,bounds_error=False,kind=kind,fill_value="extrapolate",copy=True)
    return f(xnew)

def regrid_all(x,vars,xnew,kind="cubic"):
    """Apply regrid to a list of variables"""
    return [regrid(x,var,xnew,kind) for var in vars]

def shallow_layer_approximation(layer_depth,age):
    """Shallow Layer Approximation of SMB paramters from layer elevations (Waddington et al. 2007)"""
    return layer_depth.copy()/age

def local_layer_approximation(layer_depth,total_thickness,age):
    """Local Layer Approximation of SMB paramters from layer elevations (Waddington et al. 2007)"""
    return -np.log(1-layer_depth/total_thickness)*total_thickness/age

def splinesmooth(x,y,smoothness=None):
    """Smooth a function using a spline"""
    nan_mask = np.isnan(y,dtype=bool)
    new_x = x[~nan_mask]
    new_y = y[~nan_mask]

    spl = UnivariateSpline(new_x, new_y,s = smoothness)
    return new_x,spl(new_x)

