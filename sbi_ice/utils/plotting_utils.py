import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sbi_ice.utils.modelling_utils import regrid,regrid_all,shallow_layer_approximation,local_layer_approximation

from tueplots import cycler,axes as tue_axes,fontsizes,figsizes
from tueplots.constants import markers
from tueplots.constants.color import palettes


PATH = os.path.dirname(os.path.abspath(__file__))


color_opts = {
    "colors": {
        "prior": "#069d3f",       #green prior
        "posterior": "#140289",    #blue posterior
        "observation": "#a90505", #red for observations/true values
        "boundary_condition": "#825600", #brown for boundary conditions
        "contrast1": "#808080",
        "contrast2": "#000000",
    },
    "color_maps":{
        "ice": mpl.cm.get_cmap("YlGnBu"),
        "age": mpl.cm.get_cmap("magma"),
        "prior_pairplot": mpl.cm.get_cmap("Blues"),
        "posterior_pairplot": mpl.cm.get_cmap("Reds"),
        "noise": mpl.cm.get_cmap("tab10")
    },
    "color_cycles":{
        "standard": plt.rcParams['axes.prop_cycle'].by_key()['color'],
    }
}

prior_alpha = 0.175
posterior_alpha = 0.2
samples_alpha = 0.15

def setup_plots(**kwargs):
    """Set .mplystyle for paper, and return color scheme defined in color_opts."""
    style = kwargs.get("style","standard")
    plt.style.use(PATH + os.sep + style + ".mplstyle")
    return color_opts

def plot_layers(x,bs,ss,dsum_iso,age_iso,real_layers=None,trackers = None,ax = None,**kwargs):
    """
    Plot the layers of an ice shelf, and its surface and base.

    Parameters:
    x (array-like): x-coordinates of the ice shelf layers.
    bs (array-like): Bottom surface elevation of the ice shelf.
    ss (array-like): Sea surface elevation of the ice shelf.
    dsum_iso (array-like): Array of thicknesses of each isochronal layer.
    age_iso (array-like): Array of ages of each isochronal layer.
    real_layers (array-like, optional): Array of real layers to be plotted.
    trackers (array-like, optional): Array of trackers to be plotted.
    ax (matplotlib.axes.Axes, optional): Axes object to plot the layers on.
    **kwargs: Additional keyword arguments for customization.

    Returns:
    None
    """

    norm = mpl.colors.Normalize(vmin=age_iso[:].min(), vmax=age_iso[:].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=color_opts["color_maps"]["age"])
    ms = mpl.markers.MarkerStyle(marker="d",fillstyle="none")
    if "shelf_color" not in kwargs:
        shelf_color = "k"
    else:
        shelf_color = kwargs["shelf_color"]
    if "linestyle" not in kwargs:
        ls = "dashed"
    else:
        ls = kwargs["linestyle"]
    ax.plot((x-x[0])/1e3,bs,shelf_color,linewidth=1.5,zorder=5)
    ax.plot((x-x[0])/1e3,ss,shelf_color,linewidth=1.5,zorder=5)
    if "no_fill" not in kwargs: 
        ax.fill_between((x-x[0])/1e3,ss,bs,color=shelf_color,alpha=0.075,linewidth=0.0)
    if "color" not in kwargs:
        for iz in range(0,dsum_iso.shape[-1]):
            ax.plot((x-x[0])/1e3,bs+dsum_iso[:,iz],linestyle=ls,linewidth=1.0,c =cmap.to_rgba(age_iso[iz]),zorder=0)
    else:
        for iz in range(0,dsum_iso.shape[-1]):
            ax.plot((x-x[0])/1e3,bs+dsum_iso[:,iz],linestyle=ls,linewidth=1.0,c ="k",zorder=0)

    if real_layers is not None:
        for layer in real_layers:
            ax.plot((x-x[0])/1e3,layer,color=color_opts["colors"]["observation"],linewidth=1.5,zorder=5)

    if trackers is not None:
        trackers = trackers[trackers[:,0] <x[-1]-1e-3]
        tr, ind = np.unique(trackers[:,0],return_index=True)
        trackers_highres = regrid(tr,trackers[ind,2],x)
        trackers_highres = np.clip(trackers_highres,bs,ss)

        #axs[1].scatter((trackers[:,0]-x[0])/1e3,trackers[:,2],s=75,marker="*",color=color_opts["colors"]["boundary_condition"])
        ax.fill_between((x-x[0])/1e3,trackers_highres,ss,color=color_opts["colors"]["boundary_condition"],alpha=0.5)


def animate_sim(geom,smb_regrid,bmb_regrid,anim_dsum_isos,anim_age_isos,anim_trackers,n_frames,dt,layer_sparsity = 5,strt_idx=1):
    """
    Generate animation of simulator from set of values of layer thicknesses and ages for each frame.
    """
    plt.rcParams.update(figsizes.icml2022_full(height_to_width_ratio=1.5))

    total_time = anim_age_isos[-1].max() - anim_age_isos[0].min()
    n_frames_iter = (int((total_time)/dt)+1)//n_frames+1

    dsum_iso = anim_dsum_isos[0,:,:,:]
    age_iso = anim_age_isos[0,:]
    trackers = anim_trackers[0]
    n_layers = dsum_iso.shape[-1]

    lines = {}
    norm = mpl.colors.Normalize(vmin=0, vmax=total_time)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=color_opts["color_maps"]["age"])

    ms = mpl.markers.MarkerStyle(marker='d',fillstyle='none')
    fig,axs = plt.subplots(3,1,sharex=True,constrained_layout=True,gridspec_kw = {"height_ratios":[1,3,1]})
    axs[1].plot((geom.x-geom.x[0])/1e3,geom.bs.flatten(),"k-",linewidth=3,zorder=5)
    axs[1].plot((geom.x-geom.x[0])/1e3,geom.ss.flatten(),"k-",linewidth=3,zorder=5)
    axs[1].fill_between((geom.x-geom.x[0])/1e3,geom.ss.flatten(),geom.bs.flatten(),color="black",alpha=0.3)

    for iz in range(dsum_iso.shape[2]//2+5,dsum_iso.shape[2],layer_sparsity):
        line = axs[1].plot((geom.x-geom.x[0])[strt_idx:]/1e3,geom.bs[strt_idx:,0]+dsum_iso[strt_idx:,0,iz],linestyle='dashed',c =cmap.to_rgba(age_iso[iz]),zorder=0)
        lines[iz] = line[0]

    fig.colorbar(cmap,ax=axs[1],label ='layer age [a]')

    title = axs[1].set_title('Layers at t=0 years')
    axs[1].set_ylabel('Elevation [m.a.s.l]')
    axs[0].plot((geom.x-geom.x[0])[strt_idx:]/1e3,smb_regrid[strt_idx:],color="C1")
    axs[0].set_ylabel(r"$\dot{a}$ [m/a]")
    axs[2].plot((geom.x-geom.x[0])[strt_idx:]/1e3,-bmb_regrid[strt_idx:],color="C1")
    axs[2].set_xlabel('Distance [km]')
    axs[2].set_ylabel(r"$\dot{b}$ [m/a]")


    def animate(frame):
        if frame < n_frames:
            t = str(frame*dt*n_frames_iter)
            title.set_text('Layers at t='+t+' years')
            dsum_iso = anim_dsum_isos[frame,:,:,:]
            age_iso = anim_age_isos[frame,:]
            for key in lines.keys():
                lines[key].set_ydata(geom.bs[strt_idx:,0]+dsum_iso[strt_idx:,0,key])
                lines[key].set_c(cmap.to_rgba(age_iso[key]))
        else:
            key = n_layers-3
        
    anim = FuncAnimation(fig, animate, frames = n_frames+1, interval=200)
    return anim



def plot_isochrones_1d(x,bs,ss,dsum_iso,age_iso,bmb_regrid,smb_regrid,real_layers=None,trackers = None):
    """
    Plot the SMB and BMB values above and below the simulation outputs.

    Inputs:
    x - Array of x values to plot on (in metres, then converted to km away from Grounding Line)
    bs - Base Elevation at x-coordinates (metres)
    ss - Surface Elevation at x-coordinates (metres)
    dsum_iso - Cumulative sum of layer thickness from bottom to top (metres)
    age_iso - Age of layers in dsum_iso - need to update this to 1D array as layer ages are constant in space by definition.
    bmb_regrid - Value of basal mass balance at x-coordinates (metres per year, negative = melting)
    smb_regrid - Value of surface accumulation at x-coordinates (metres per year, positive = gaining mass)
    trackers - (x,z) coordinates of points to plot
    real_layers - array of (x,z) coordinates for each real layer to plot
    """
    norm = mpl.colors.Normalize(vmin=age_iso[:].min(), vmax=age_iso[:].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=color_opts["color_maps"]["age"])

    ms = mpl.markers.MarkerStyle(marker="d",fillstyle="none")
    fig,axs = plt.subplots(3,1,sharex=True,constrained_layout=True,gridspec_kw = {"height_ratios":[1,3,1]})

    axs[1].plot((x-x[0])/1e3,bs,"k-",linewidth=1.5,zorder=5)
    axs[1].plot((x-x[0])/1e3,ss,"k-",linewidth=1.5,zorder=5)
    axs[1].fill_between((x-x[0])/1e3,ss,bs,color="black",alpha=0.075,linewidth=0.0)

    for iz in range(0,dsum_iso.shape[-1]):
        axs[1].plot((x-x[0])/1e3,bs+dsum_iso[:,iz],linewidth=1.0,linestyle="dashed",c =cmap.to_rgba(age_iso[iz]),zorder=0)

    if real_layers is not None:
        for layer in real_layers:
            axs[1].plot((x-x[0])/1e3,layer,color=color_opts["colors"]["observation"],linewidth=1.5,zorder=5)

    if trackers is not None:
        trackers = trackers[trackers[:,0] <x[-1]-1e-3]
        tr, ind = np.unique(trackers[:,0],return_index=True)
        trackers_highres = regrid(tr,trackers[ind,2],x)
        trackers_highres = np.clip(trackers_highres,bs,ss)

        axs[1].fill_between((x-x[0])/1e3,trackers_highres,ss,color=color_opts["colors"]["boundary_condition"],alpha=0.3,linewidth=0.0)

    fig.colorbar(cmap,ax=axs[1],label ="layer age [a]")


    axs[1].set_title("Layers")
    axs[1].set_ylabel("elevation [m.a.s.l]")


    axs[0].plot((x-x[0])/1e3,smb_regrid,color=color_opts["colors"]["prior"])
    axs[0].set_title("surface accummulation")
    axs[0].set_ylabel(r"$\dot{a}$ [m/a]")

    axs[2].plot((x-x[0])/1e3,-bmb_regrid,color=color_opts["colors"]["prior"])
    axs[2].set_title("basal melting")
    axs[2].set_ylabel(r"$\dot{b}$ [m/a]")
    axs[2].set_xlabel("Distance [km]")

    for ax in axs[:-1]:
        ax.tick_params(
            axis='x',          
            which='both',      
            bottom=False,      
        )


    return fig,axs

def plot_posterior_nice(x,
                        mb_mask,
                        tmb,
                        prior_smb_samples,
                        posterior_smb_samples,
                        layer_mask,
                        LMI_boundary,
                        prior_layer_samples,
                        prior_layer_ages,
                        posterior_layer_samples,
                        posterior_layer_ages,
                        true_layer,
                        shelf_base,
                        shelf_surface, 
                        true_smb = None,
                        true_age = None,
                        plot_samples = False,
                        title = None):
    """
    Plot the posterior SMB (top) and BMB (bottom) values, along with the posterior predictive (layers, middle)

    Args:
        x (ndarray): Discretization of flowline domain.
        mb_mask (ndarray): Boolean mask indicating the locations where SMB is inferred
        tmb (ndarray): Fixed total mass balance values [m/a].
        prior_smb_samples (ndarray): Prior SMB samples.
        posterior_smb_samples (ndarray): Posterior SMB samples.
        layer_mask (ndarray): Boolean mask indicating the locations where the layer elevations were used for training.
        LMI_boudnary (float): LMI boundary for this layer.
        prior_layer_samples (ndarray): Prior layer samples.
        prior_layer_ages (ndarray): Prior layer ages.
        posterior_layer_samples (ndarray): Posterior layer samples.
        posterior_layer_ages (ndarray):  Posterior layer ages.
        true_layer (ndarray): True layer elevations.
        shelf_base (ndarray): Shelf base elevations.
        shelf_surface (ndarray): Shelf surface elevations.
        true_smb (ndarray, optional): True SMB values.
        true_age (float, optional): Age of GT layer (if known).
        plot_samples (bool, optional): Flag indicating whether to plot individual samples. Defaults to False.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    if true_age is not None:
        x_label = "Distance along flowline [km]"
        gt_label = "GT"
    else:
        x_label = "Distance from GL [km]"
        gt_label ="Observed"
    plt.rcParams.update(figsizes.icml2022_full(nrows=3,height_to_width_ratio=0.35))

    ax_label_loc = -0.085
    #percentiles are approx. 2sigma 
    percentiles = [5,95]
    #age percentiles are approx. 1sigma
    age_percentiles = [16,84]

    #Calculate prior and posterior mean and quantiles
    post_mean_smb = torch.mean(posterior_smb_samples,axis=0)
    post_uq_smb = torch.quantile(posterior_smb_samples,percentiles[1]/100,axis=0)
    post_lq_smb = torch.quantile(posterior_smb_samples,percentiles[0]/100,axis=0)
    prior_mean_smb = torch.mean(prior_smb_samples,axis=0)
    prior_uq_smb = torch.quantile(prior_smb_samples,percentiles[1]/100,axis=0)
    prior_lq_smb = torch.quantile(prior_smb_samples,percentiles[0]/100,axis=0)

    #Calculate prior and posterior layer elevation mean and quantiles
    post_layer_mean = np.mean(posterior_layer_samples,axis=0)
    post_uq_layer = np.quantile(posterior_layer_samples,percentiles[1]/100,axis=0)
    post_lq_layer = np.quantile(posterior_layer_samples,percentiles[0]/100,axis=0)
    prior_layer_mean = torch.mean(prior_layer_samples,axis=0)
    prior_uq_layer = torch.quantile(prior_layer_samples,percentiles[1]/100,axis=0)
    prior_lq_layer = torch.quantile(prior_layer_samples,percentiles[0]/100,axis=0)

    #Calculate prior and posterior layer age mean and quantiles
    posterior_age_median = np.quantile(posterior_layer_ages,0.5)
    posterior_age_uq = np.quantile(posterior_layer_ages,age_percentiles[1]/100)
    posterior_age_lq = np.quantile(posterior_layer_ages,age_percentiles[0]/100)


    fig,axs = plt.subplots(3,1,sharex=True,gridspec_kw = {"height_ratios":[1,4,1]})

    #First axis plots the prior and posterior SMB distributions
    ax = axs[0]
    ax.plot(x[mb_mask]/1e3,prior_mean_smb,color=color_opts["colors"]["prior"],label="Prior Mean",linewidth=1.0)
    ax.plot(x[mb_mask]/1e3,post_mean_smb,color=color_opts["colors"]["posterior"],label="Posterior Mean",linewidth=1.0)
    if plot_samples:
        for i in range(20):
            ax.plot(x[mb_mask]/1e3,posterior_smb_samples[i],color=color_opts["colors"]["posterior"],alpha=samples_alpha,linewidth=0.5)
            ax.plot(x[mb_mask]/1e3,prior_smb_samples[i],color=color_opts["colors"]["prior"],alpha=samples_alpha,linewidth=0.5)
    else:
        ax.fill_between(x[mb_mask]/1e3,prior_lq_smb,prior_uq_smb,color=color_opts["colors"]["prior"],alpha=prior_alpha,linewidth=0.0)
        ax.fill_between(x[mb_mask]/1e3,post_lq_smb,post_uq_smb,color=color_opts["colors"]["posterior"],alpha=posterior_alpha,linewidth=0.0)

    ax.set_ylabel('$\dot{a}$ [m/a]')

    #Plot real smb if possible
    if true_smb is not None:
        true_smb_mask = np.where(x < x[mb_mask][-1] +1e-3)
        ax.plot(x[true_smb_mask]/1e3,true_smb[true_smb_mask],color=color_opts["colors"]["observation"],linewidth=1.0,label="True SMB")

    ax.text(ax_label_loc, 0.95, "a", transform=ax.transAxes,fontsize=12, va='top', ha='right')

    #Second axis plots the prior and posterior predictive distributions
    ax = axs[1]
        
    #We split the axis into two to show the layers in higher detail
    split_layers = True
    if split_layers:
        divider = make_axes_locatable(ax)
        ax_ratio = 4.0
        ax2 = divider.new_vertical(size=str(100*ax_ratio)+"%", pad=0.05)
        fig.add_axes(ax2)
        ax1s = [ax,ax2]
        ax.set_ylim(np.min(shelf_base), np.max(shelf_base)+5)
        dist = np.mean(shelf_surface[layer_mask]-post_layer_mean)
        ax2.set_ylim(np.min((post_layer_mean -0.4*dist)[layer_mask]),np.max(shelf_surface))
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines[["bottom","top","right"]].set_visible(False)

        ax2.set_ylabel('Elevation [m.a.s.l]')
        ax2.yaxis.set_label_coords(ax_label_loc, 0.35, transform=ax2.transAxes)
        loc = plticker.MultipleLocator(base=20.0)
        ax.xaxis.set_major_locator(loc)
        ax2.text(ax_label_loc, 0.95, "b", transform=ax2.transAxes,fontsize=12, va='top', ha='right')


        d = .01 
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False,linewidth=1.0)
        ax2.plot((-d, +d), (-d, +d), **kwargs)       
        kwargs.update(transform=ax.transAxes) 
        ax.plot((-d, +d), (1 - d*ax_ratio, 1 + d*ax_ratio), **kwargs)

    else:
        ax1s = [ax]
    for axi in ax1s:
        #First, plot the shelf surface and base
        s1, = axi.plot(x/1e3,shelf_surface,color="black",linewidth = 1.0)
        s2, = axi.plot(x/1e3,shelf_base,color="black",linewidth = 1.0)
        s3 = axi.fill_between(x/1e3,shelf_surface,shelf_base,color="black",alpha=0.075,linewidth=0.0)
        pr1, = axi.plot(x[layer_mask]/1e3,prior_layer_mean[layer_mask],color=color_opts["colors"]["prior"],linewidth=0.8)
        po1, = axi.plot(x[layer_mask]/1e3,post_layer_mean,color=color_opts["colors"]["posterior"],zorder=5,linewidth=0.8)
        #annotate posterior layer age
        axi.annotate(r'age = ${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ a'.format(posterior_age_median,posterior_age_uq-posterior_age_median,posterior_age_median-posterior_age_lq),
                     xy=(x[layer_mask][0]/1e3,post_layer_mean[0]),xycoords="data",textcoords="offset points",
                     xytext=(0,-30),color = color_opts["colors"]["posterior"])

        if plot_samples:
            for i in range(20):
                axi.plot(x[layer_mask]/1e3,prior_layer_samples[i,layer_mask],color=color_opts["colors"]["prior"],alpha=samples_alpha,linewidth=0.5)
                axi.plot(x[layer_mask]/1e3,posterior_layer_samples[i],color=color_opts["colors"]["posterior"],alpha=samples_alpha,linewidth=0.5)
        else:
            pr2 = axi.fill_between(x[layer_mask]/1e3,prior_lq_layer[layer_mask],prior_uq_layer[layer_mask],color=color_opts["colors"]["prior"],alpha=prior_alpha,linewidth=0.0)
            po2 = axi.fill_between(x[layer_mask]/1e3,post_lq_layer,post_uq_layer,color=color_opts["colors"]["posterior"],alpha=posterior_alpha,zorder=5,linewidth=0.0)

        ob1, = axi.plot(x[layer_mask]/1e3,true_layer[layer_mask],color=color_opts["colors"]["observation"],
                    zorder=10,linewidth=1.0)

        
        if true_age is not None:
            nx = len(x[layer_mask])
            axi.annotate(r'true age = {:.0f} a'.format(true_age),
                     xy=(x[layer_mask][0]/1e3,post_layer_mean[0]),xycoords="data",textcoords="offset points",
                     xytext=(0,25),color = color_opts["colors"]["observation"])

        #Draw vertical line at LMI boundary
        axi.axvline(x=LMI_boundary/1e3,color=color_opts["colors"]["boundary_condition"],linestyle="--",label="LMI boundary")
        
        
    labels = ["Shelf","Prior","Posterior",gt_label]
    ax2.legend(handles=[(s1,s3),(pr1,pr2),(po1,po2),(ob1,)],labels = labels,bbox_to_anchor = (0.0,0,0.95,0.95),loc="upper right", ncol=2)


    #Thid axis = plot prior and posterior for BMB
    bmb_vals = (post_mean_smb-tmb[mb_mask] < 10).numpy()
    bmb_mask = np.zeros_like(mb_mask,dtype=bool)
    bmb_mask[mb_mask] = bmb_vals
    ax = axs[2]

    #Could similarly split the bmb axis into two
    split_bmb = False
    if split_bmb:
        divider = make_axes_locatable(ax)
        ax_ratio = 1.0/2.0
        ax2 = divider.new_vertical(size=str(100*ax_ratio)+"%", pad=0.05)
        fig.add_axes(ax2)
        ax1s = [ax,ax2]
        ax.set_ylim(-0.5,1.55)
        dist = np.mean(shelf_surface[layer_mask]-post_layer_mean)
        ax2.set_ylim(1.55,5.0)
        ax2.set_yticks((2,5))
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines[["bottom","top","right"]].set_visible(False)

        ax.set_ylabel('$\dot{b}$ [m/a]')
        ax.yaxis.set_label_coords(ax_label_loc+0.05, 0.85, transform=ax.transAxes)
        loc = plticker.MultipleLocator(base=20.0)
        ax.xaxis.set_major_locator(loc)
        ax2.text(ax_label_loc, 1.2, "c", transform=ax2.transAxes,fontsize=12, va='top', ha='right')


        d = .01 
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False,linewidth=1.0)
        ax2.plot((-d, +d), (-10*d, +10*d), **kwargs)     
        kwargs.update(transform=ax.transAxes)
        ax.plot((-d, +d), (1 - 10*d*ax_ratio, 1 + 10*d*ax_ratio), **kwargs)
        ax.set_xlabel(x_label)
        ax.spines['bottom'].set_bounds(x[0]/1e3-0.001,x[-1]/1e3)



    else:
        ax1s = [ax]
        ax.set_ylabel('$\dot{b}$ [m/a]')

        ax.set_xlabel(x_label)
        ax.text(ax_label_loc, 0.95, "c", transform=ax.transAxes,fontsize=12, va='top', ha='right')
        ax.spines['bottom'].set_bounds(x[0]/1e3-0.001,x[-1]/1e3)
        # ax.set_ylim(-0.5,1.5)
        # ax.set_yticks((0.0,1.0))

    for ax in ax1s:
        ax.plot(x[bmb_mask]/1e3,prior_mean_smb[bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["prior"],label="Prior",linewidth=1.0)
        ax.plot(x[bmb_mask]/1e3,post_mean_smb[bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["posterior"],label="Posterior",linewidth=1.0)
        if plot_samples:
            for i in range(20):
                ax.plot(x[bmb_mask]/1e3,posterior_smb_samples[i][bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["posterior"],alpha=samples_alpha,linewidth=0.5)
                ax.plot(x[bmb_mask]/1e3,prior_smb_samples[i][bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["prior"],alpha=samples_alpha,linewidth=0.5)
        else:
            ax.fill_between(x[bmb_mask]/1e3,prior_lq_smb[bmb_vals]-tmb[bmb_mask],prior_uq_smb[bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["prior"],alpha=prior_alpha,linewidth=0.0)
            ax.fill_between(x[bmb_mask]/1e3,post_lq_smb[bmb_vals]-tmb[bmb_mask],post_uq_smb[bmb_vals]-tmb[bmb_mask],color=color_opts["colors"]["posterior"],alpha=posterior_alpha,linewidth=0.0)

        

        #Plot real bmb if possible
        if true_smb is not None:
            true_bmb = true_smb-tmb
            ax.plot(x[true_smb_mask]/1e3,true_bmb[true_smb_mask],color=color_opts["colors"]["observation"],linewidth=1.0,label="True BMB")



    for ax in axs[:-1]:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
        )
        ax.spines[["bottom","top","right"]].set_visible(False)
    return fig,axs


def plot_posterior_spatial( x,
                            mb_mask,
                            tmb,
                            prior_smb_samples,
                            posterior_smb_samples,
                            true_smb = None,
                            plot_samples = True):
    
    """Plot prior and posterior SMB and BMB values."""
    plt.rcParams.update(figsizes.icml2022_full(nrows=2,height_to_width_ratio=1.0))

    fig,axs = plt.subplots(2,1,sharex=True)
    percentiles = [5,95]

    post_mean_smb = torch.mean(posterior_smb_samples,axis=0)
    post_uq_smb = torch.quantile(posterior_smb_samples,percentiles[1]/100,axis=0)
    post_lq_smb = torch.quantile(posterior_smb_samples,percentiles[0]/100,axis=0)
    prior_mean_smb = torch.mean(prior_smb_samples,axis=0)
    prior_uq_smb = torch.quantile(prior_smb_samples,percentiles[1]/100,axis=0)
    prior_lq_smb = torch.quantile(prior_smb_samples,percentiles[0]/100,axis=0)

    #First axis plots the prior and the posterior in parameter space
    ax = axs[0]
    ax.plot(x[mb_mask]/1e3,prior_mean_smb,color=color_opts["colors"]["prior"],marker="o",ms=2,label="Prior Mean")
    ax.fill_between(x[mb_mask]/1e3,prior_lq_smb,prior_uq_smb,color=color_opts["colors"]["prior"],alpha=0.2)
    ax.plot(x[mb_mask]/1e3,post_mean_smb,color=color_opts["colors"]["posterior"],marker="o",ms=2,label="Posterior Mean")
    ax.fill_between(x[mb_mask]/1e3,post_lq_smb,post_uq_smb,color=color_opts["colors"]["posterior"],alpha=0.2)
    if plot_samples:
        for i in range(5):
            ax.plot(x[mb_mask]/1e3,posterior_smb_samples[i],color=color_opts["colors"]["posterior"],alpha=0.4)
            ax.plot(x[mb_mask]/1e3,prior_smb_samples[i],color=color_opts["colors"]["prior"],alpha=0.4)
    ax.set_ylabel(r"$\dot{a}$ [m/a]")
    #Plot real smb if possible
    if true_smb is not None:
        ax.plot(x/1e3,true_smb,color=color_opts["colors"]["observation"],linewidth=2,linestyle="dashed",label="True SMB")

    #Second axis = plot prior and posterior for BMB
    ax = axs[1]
    ax.plot(x[mb_mask]/1e3,prior_mean_smb-tmb[mb_mask],color=color_opts["colors"]["prior"],marker="o",ms=2,label="Prior")
    ax.fill_between(x[mb_mask]/1e3,prior_lq_smb-tmb[mb_mask],prior_uq_smb-tmb[mb_mask],color=color_opts["colors"]["prior"],alpha=0.2)
    ax.plot(x[mb_mask]/1e3,post_mean_smb-tmb[mb_mask],color=color_opts["colors"]["posterior"],marker="o",ms=2,label="Posterior")
    ax.fill_between(x[mb_mask]/1e3,post_lq_smb-tmb[mb_mask],post_uq_smb-tmb[mb_mask],color=color_opts["colors"]["posterior"],alpha=0.2)
    if plot_samples:
        for i in range(5):
            ax.plot(x[mb_mask]/1e3,posterior_smb_samples[i]-tmb[mb_mask],color=color_opts["colors"]["posterior"],alpha=0.4)
            ax.plot(x[mb_mask]/1e3,prior_smb_samples[i]-tmb[mb_mask],color=color_opts["colors"]["prior"],alpha=0.4)
    ax.set_ylabel(r"$\dot{b}$ [m/a]")
    ax.set_xlabel("Distance [km]")
    #Plot real bmb if possible
    if true_smb is not None:
        true_bmb = true_smb-tmb
        ax.plot(x/1e3,true_bmb,color=color_opts["colors"]["observation"],linewidth=2,linestyle="dashed",label="True BMB")
    ax.set_xlim(x.min()/1e3-2,x.max()/1e3+2)
    #ax.set_ylim(-1.0,2.5)

    for ax in axs[:-1]:
            ax.tick_params(
                axis='x',
                which='both',
                bottom=False,
            )

    return fig,axs

def compare_posteriors(x,posteriors,layers,real_smb=None,kottas_smb = None,labels = None,prior=None):
    """
    Compare posteriors for different layers with validation methods

    Parameters:
    x (array-like): The x-coordinates for the plot.
    posteriors (list): A list of posterior distributions for each layer.
    layers (list): A list of layer information dictionaries.
    real_smb (array-like, optional): The true SMB values. Default is None.
    kottas_smb (dict, optional): The Kottas SMB values. Default is None.
    labels (list, optional): A list of labels for each layer. Default is None.
    prior (array-like, optional): The prior distribution. Default is None.

    Returns:
    fig (matplotlib.figure.Figure): The generated figure.
    axs (numpy.ndarray): The array of Axes objects.
    """
    
    #Currently implemented only for exactly 4 layers.
    assert len(posteriors) == 4 and len(layers) == 4
    plt.rcParams.update(figsizes.icml2022_full(nrows=2,ncols=2))
    fig,axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)

    for axi in axs.flatten():
        axi.hlines(0.0,x[0],x[-1],color="black",linestyle="--")
    if labels is None:
        labels = ["Posterior SMB for layer " + str(i+1) for i in range(len(posteriors))]


    for idx,fname in enumerate(posteriors):
        #Calculate mean and quantiles of distributions.
        posterior = posteriors[idx]
        posterior_samples = posterior.sample((1000,))
        percentiles = [5,95]

        post_mean_smb = torch.mean(posterior_samples,axis=0)
        post_uq_smb = torch.quantile(posterior_samples,percentiles[1]/100,axis=0)
        post_lq_smb = torch.quantile(posterior_samples,percentiles[0]/100,axis=0)
        #post_std_smb = torch.std(posterior_samples,axis=0)
        ax = axs[idx//2,idx%2]
        po1, = ax.plot(x,post_mean_smb,color = color_opts["colors"]["posterior"],linewidth = 1.0)
        po2 = ax.fill_between(x,post_uq_smb,post_lq_smb,alpha=posterior_alpha,color = color_opts["colors"]["posterior"],linewidth=0.0)
        layer_x = layers[idx]["layer_x"]
        LMI_x = layers[idx]["LMI_x"]
        layer_depth = layers[idx]["layer_depth"]
        layer_age = layers[idx]["layer_age"]
        total_thickness = layers[idx]["total_thickness"]
        ax.set_title("IRH {} ({:.0f}a)".format(idx+1,layer_age),x=0.1 if idx%2 == 0 else 0.6)
        ax.vlines(x=LMI_x[0],ymin =0.05, ymax=0.95, transform=ax.get_xaxis_transform(),color=color_opts["colors"]["boundary_condition"],
                  linestyle="--")
        
        #Calculate SLA and LLA approximations
        SLA = shallow_layer_approximation(layer_depth=layer_depth,age=layer_age)
        LLA = local_layer_approximation(layer_depth=layer_depth,total_thickness=total_thickness,age=layer_age)
        sl1, = ax.plot(layer_x[layer_x<x[-1]+1e-5],SLA[layer_x<x[-1]+1e-5],color = color_opts["colors"]["contrast1"],linestyle="solid", linewidth=1.2)
        ll1, = ax.plot(layer_x[layer_x<x[-1]+1e-5],LLA[layer_x<x[-1]+1e-5],color = color_opts["colors"]["contrast2"],linestyle="solid", linewidth=1.2, zorder=-1)
        for axi in axs.flatten():
            if axi != ax:
                # axi.plot(x,post_mean_smb,color = "black",alpha=0.25)
                # axi.fill_between(x,post_uq_smb,post_lq_smb,alpha=0.1,color = "black")
                continue

    #If either the GT or validation data is given, plot it.
    if real_smb is not None:
        for axi in axs.flatten():
            gt1, = axi.plot(layer_x[layer_x<x[-1]+1e-5],real_smb[layer_x<x[-1]+1e-5],color=color_opts["colors"]["observation"],linewidth = 1.2)
    if kottas_smb is not None:
        for axi in axs.flatten():
            kt1, = axi.plot(kottas_smb["kottas_xmb"]/1e3,
                    kottas_smb["kottas_time_mean_smb"],
                    color=color_opts["colors"]["observation"],
                    linewidth=1.0,
                    )

    for i in range(2):
        for j in range(2):
            if i==1:
                axs[i,j].spines['bottom'].set_bounds(x[0]-0.001,x[-1]+2)
            elif i==0:
                axs[i,j].spines[['bottom']].set_visible(False)
                axs[i,j].tick_params(axis='x', which='both',bottom=False)
            if j==0:
                pass
            elif j==1:
                axs[i,j].spines[['left']].set_visible(False)
                axs[i,j].tick_params(axis='y', which='both',left=False)

    fig.supxlabel("Distance from GL [km]")
    fig.supylabel("Surface accumulation [m/a]")

    handles = [(po1,po2),(sl1,),(ll1,)]
    labels = ["SBI Posterior","SLA","LLA"]
    if real_smb is not None:
        handles.append((gt1,))
        labels.append("True SMB")
    if kottas_smb is not None:
        handles.append((kt1,))
        labels.append("Kottas SMB")
    fig.legend(handles =handles,labels=labels, loc="center",ncol=2)
    return fig,axs


