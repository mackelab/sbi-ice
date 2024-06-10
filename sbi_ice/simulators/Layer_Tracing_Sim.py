import numpy as np
import pandas as pd
from sbi_ice.utils.modelling_utils import regrid,regrid_all,trunc

#advection tolerances for superbee advection
adv_eps1 = 1e-9
adv_eps2 = 1e-9

def read_setup(fname):
    """
    Load data from csv into numpy arrays

    Arguments:
    fname: filename of csv file to use
    """
    df = pd.read_csv(fname)
    df = df.sort_values(by = 'x_coord')
    x = df['x_coord'].to_numpy() #x - coordinates of domain
    bs = df['base'].to_numpy() #base z- coordinates
    ss =df['surface'].to_numpy() #surface z- coordinates
    vxs = df['velocity'].to_numpy() #x-velocities
    tmb = df['tmb'].to_numpy() #spatial surface mass balance

    #If Flux gradients are available, load them as well
    try:
        dQdy = df['dQdy'].to_numpy()
        dQdx = df['dQdx'].to_numpy()
    except:
        dQdy = None
        dQdx = None

    return x,bs,ss,vxs,tmb,dQdx,dQdy


class Tracker():
    "Class to contain particle tracker positions"
    def __init__(self,x=0,layer=0,z=0):
        """
        Arguments:
        x: x-position of particle
        layer: layer index tracker by particle
        """
        self.x = x
        self.layer = layer
        self.z = z

    def update_layer(self,layer):
        """
        Update the layer index of the particle

        Arguments:
        layer: new layer index
        """
        self.layer = layer

    def update_x(self,x):
        """
        Update the x-position of the particle

        Arguments:
        x: new x-position
        """
        self.x = x

    def update_z(self,z):
        """
        Update the z-position of the particle

        Arguments:
        z: new z-position
        """
        self.z = z
    def is_active(self,layers):
        """
        Check if the particle is still in the domain

        Arguments:
        layers: number of layers in the domain
        """
        if self.layer in layers:
            return True
        else:
            return False
        
class Trackers():
    "Class to contain multiple Tracker objects"
    def __init__(self,trackers = []):
        self.trackers = trackers

    def add_tracker(self,tracker):
        """
        Add a new tracker to the list

        Arguments:
        x: x-position of particle
        layer: layer index tracker by particle
        """
        self.trackers.append(tracker)

    
    def remove_inactive(self,layers):
        """
        Remove inactive trackers from the list

        Arguments:
        layers: number of layers in the domain
        """
        self.trackers = [tracker for tracker in self.trackers if tracker.is_active(layers)]

    def relabel_layers(self,old_layers,new_layers):
        """
        Relabel the layers of the trackers

        Arguments:
        old_layers: the old layer indices of the layers to keep
        new_layers: the new layer indices
        """
        self.remove_inactive(old_layers)
        for tracker in self.trackers:
            tracker.update_layer(new_layers[np.where(old_layers==tracker.layer)[0][0]])


class Scheduele():
    "Class to contain the time stepping scheduele"
    def __init__(self,sim_time,dt,n_iter_surface,n_iter_base=None):
        """
        Arguments:
        time_init: initial time
        time_end: final time
        dt: time step
        n_iter_surface: number of iterations per addition of surface layer
        n_iter_base: number of iterations per addition of base layer
        """
        self.sim_time = sim_time

        self.dt = dt
        self.n_iter_surface = n_iter_surface
        self.n_iter_base = n_iter_base if n_iter_base is not None else n_iter_surface
        self.it = 0
        self.n_layer_base = int((self.sim_time/(self.n_iter_base*self.dt)))
        self.n_layer_surface = int((self.sim_time/(self.n_iter_surface*self.dt)))
        self.total_iterations = int(self.sim_time/self.dt)

    def bool_surface(self):
        """
        Check if a surface layer should be added
        """
        return self.it%self.n_iter_surface == 0

    def bool_base(self):
        """
        Check if a base layer should be added
        """
        return self.it%self.n_iter_base == 0
    def update(self):
        """
        Update the iteration count
        """
        self.it += 1
        


class Geom():
    """
    Base Class to contain the Born simulator objects, mainly the layer elevations
    """
    def __init__(self, nx_iso,ny_iso):
        """
        Arguments:

        nx_iso: number of evaluation points in x-direction
        ny_iso: number of evaluation pointsin y-direction
        init_layers: number of initial non-0 layers (placeholders)
        """
        self.nx = nx_iso
        self.ny = ny_iso
        self.elapsed_t = 0

    def fields_from_arrays(self,x_setup,base,surface,velocity,dQdx=None,dQdy=None):
        """
        Initialize the domain and dynamical variables of Born model

        x_setup: original x domain [metres] given in data
        base: the elevation of the ice base (bottom) [metres] as evaluated at x_setup
        surface: the elevation of the ice surface (top) [metres] as evaluated at x_setup
        velocity: the x-velocity of the ice [metres/year] as evaluated at x_setup (plug flow - constant in depth)
        dQdx: the x-derivative of the ice flux, Q, evaluated at x_setup [1/year]
        dQdy: the y-derivative of the ice flux, Q, evaluated at x_setup [1/year]
        """
        self.x = np.linspace(x_setup[0],x_setup[-1],self.nx)
        self.dx = self.x[1]-self.x[0]
        self.Lx = self.x[-1] - self.x[0]
        self.vxs = regrid(x_setup,velocity,np.linspace(self.x[0]-self.dx/2,self.x[-1]+self.dx/2,self.nx+1,endpoint=True)).reshape(self.nx+1,self.ny)
        self.vxs2 = regrid(x_setup,velocity,np.linspace(self.x[0],self.x[-1],self.nx+1)).reshape(self.nx+1,self.ny)

        self.ss = regrid(x_setup,surface,self.x).reshape(self.nx,self.ny)
        self.bs = regrid(x_setup,base,self.x).reshape(self.nx,self.ny)

        self.dQdx = regrid(x_setup,dQdx,self.x).reshape(self.nx,self.ny)
        self.dQdy = regrid(x_setup,dQdy,self.x).reshape(self.nx,self.ny)


    def initialize_layers(self, sched:Scheduele, init_nlayers:int, tracker_coords:np.ndarray= np.array([[0.0,0.0]])):
        """
        Initialize arrays to hold layer thickness, layer age, particle tracker positions.
        Most layers start at thickness 0.0, the middle init_layers are equal in thicknesses to add up to the total ice shelf thickness

        Arguments:
        sched: the time stepping scheduele
        init_nlayers: number of initial non-0 layers (placeholders)
        """


        if hasattr(self, 'd_iso'):
            temp = self.d_iso[:,:,self.layer_mask].copy()
            del self.d_iso
            self.d_iso = temp
            self.age_iso = self.age_iso[self.layer_mask]
            self.trackers.relabel_layers(self.layer_mask,np.arange(self.layer_mask.size))
            self.layer_mask = np.arange(self.layer_mask.size)
            self.top_layer = self.layer_mask[-1]
            self.bottom_layer = 0

        else:
            self.layer_mask = np.arange(init_nlayers)
            self.top_layer = self.layer_mask[-1]
            self.bottom_layer = 0

            h_ice = self.ss-self.bs #ice thickness
            H_ice_iso = h_ice/init_nlayers #thickness of equally split initial layers
            self.d_iso = np.repeat(H_ice_iso[:, :, np.newaxis], init_nlayers, axis=2) #thickness of each individual layer 
            d_iso_out = np.zeros(shape=(self.nx,self.ny,init_nlayers)) #temporary layer thickness

            sum_iz = np.sum(self.d_iso[:,:,self.bottom_layer:self.top_layer+1],axis=2)
            for iz in range(self.bottom_layer,self.top_layer+1):
                d_iso_out[:,:,iz] = self.d_iso[:,:,iz]*h_ice/sum_iz
            self.d_iso  = d_iso_out
            self.dsum_iso = np.cumsum(self.d_iso,axis=2)
            self.age_iso = np.zeros(init_nlayers)
            self.tracker_coords = tracker_coords
            tracker_positions = []
            for pos in self.tracker_coords:
                tracker_x = np.argmin(np.abs(self.x-pos[0]))
                tracker_z = np.argmin(np.abs((self.ss[tracker_x,0]-(self.dsum_iso[tracker_x,0,:]+self.bs[tracker_x,0]))-pos[1]))
                tracker_z = self.top_layer-tracker_z
                print(tracker_z)
                tracker_positions.append([tracker_x,tracker_z])
            self.trackers = Trackers([Tracker(self.x[pos[0]],self.top_layer-pos[1],self.dsum_iso[pos[0],0,self.top_layer-pos[1]]) for pos in tracker_positions])


        temp = np.concatenate((np.zeros(shape=(self.nx,self.ny,sched.n_layer_base)),self.d_iso,np.zeros(shape=(self.nx,self.ny,sched.n_layer_surface))),axis=2).copy()
        del self.d_iso
        self.d_iso = temp
        self.age_iso = np.concatenate((np.zeros(sched.n_layer_base),self.age_iso,np.zeros(sched.n_layer_surface)))
        self.trackers.relabel_layers(self.layer_mask,self.layer_mask+sched.n_layer_base)
        self.layer_mask += sched.n_layer_base
        self.top_layer += sched.n_layer_base
        self.bottom_layer += sched.n_layer_base
        self.dsum_iso = np.cumsum(self.d_iso,axis=2)




    def extract_nonzero_layers(self):
        """
        Extract the thickness and age of all non-zero layers
        """

        return self.layer_mask,self.dsum_iso[:,:,self.layer_mask],self.age_iso[self.layer_mask]

    def extract_active_trackers(self):
        """
        Extract the positions of all active trackers
        """
        assert len(self.trackers.trackers)>0, "There are no active trackers"

        self.trackers.remove_inactive(self.layer_mask)
        pos_array = np.zeros(shape=(len(self.trackers.trackers),3))
        for i,tracker in enumerate(self.trackers.trackers):
            pos_array[i,:] = np.array([tracker.x,0,self.dsum_iso[np.argmin(np.abs(self.x-tracker.x)),0,tracker.layer]+self.bs[np.argmin(np.abs(self.x-tracker.x)),0]])


        return pos_array[pos_array[:,0]<self.x[-1],:]



def init_geom_from_fname(geom,setup_fname,regrid_x = None,**kwargs):
    """
    Initialize fields of a Geom object directly from a setup file.

    geom: Geom object to initialize
    setup_fname: file name holding domain and dynamical variables
    regrid_x: new x-grid to use for domain (if not using one stored in the setup file)
    """
    x_setup,bs,ss,vxs,tmb,dQdx,dQdy = init_fields_from_fname(geom,setup_fname,regrid_x)
    smb_regrid,bmb_regrid = init_mb(geom,x_setup,tmb,**kwargs)
    return smb_regrid,bmb_regrid

def init_fields_from_fname(geom,setup_fname,regrid_x = None):
    """
    Initialize fields of a Geom object directly from a setup file.

    geom: Geom object to initialize
    setup_fname: file name holding domain and dynamical variables
    regrid_x: new x-grid to use for domain (if not using one stored in the setup file)
    """
    x_setup,bs,ss,vxs,tmb,dQdx,dQdy = read_setup(setup_fname)
    if regrid_x is not None:
        bs,ss,vxs,tmb = regrid_all(x_setup,[bs,ss,vxs,tmb,dQdx,dQdy],regrid_x)
        x_setup = regrid_x
    geom.fields_from_arrays(x_setup,bs,ss,vxs,dQdx,dQdy)
    return x_setup,bs,ss,vxs,tmb,dQdx,dQdy

def init_from_fields(geom,x_setup,bs,ss,vxs,tmb,dQdx,dQdy,regrid_x=None,**kwargs):
    """
    Initialize fields of a Geom object

    See fields_from_arrays and init_from_fname for arguments
    """
    if regrid_x is not None:
        bs,ss,vxs,tmb = regrid_all(x_setup,[bs,ss,vxs,tmb,dQdx,dQdy],regrid_x)
        x_setup = regrid_x
    geom.fields_from_arrays(x_setup,bs,ss,vxs,dQdx,dQdy)
    smb_regrid,bmb_regrid = init_mb(geom,x_setup,tmb,**kwargs)
    return smb_regrid,bmb_regrid

def init_mb(geom,xmb,tmb=None,smb = None,bmb = None):
    """
    Initialize the surface and basal mass balance arrays we can use for simulating for a given Geom object

    Arguments:
    geom: Born Geom() object we want to simulate
    xmb: x-array that the given mass balance is defined from
    tmb: total mass balance - can be used to calculate smb/bmb from the other if tmb is a known constant
    smb: surface mass balance array
    bmb: basal mass balance array
    """

    #if both smb and bmb are given, just regrid them onto geom.x and we are done
    if smb is not None and bmb is not None:
        smb_regrid =  regrid(xmb,smb,geom.x).reshape(geom.nx,geom.ny)
        bmb_regrid =  regrid(xmb,bmb,geom.x).reshape(geom.nx,geom.ny)
    #otherwise, figure out what is defined, and calculate the other using total mass balance
    else:
        assert tmb is not None
        tmb_regrid =  regrid(xmb,tmb,geom.x).reshape(geom.nx,geom.ny)
        if smb is not None:
            smb_regrid =  regrid(xmb,smb,geom.x).reshape(geom.nx,geom.ny)
            bmb_regrid = smb_regrid - tmb_regrid
        elif bmb is not None:
            bmb_regrid =  regrid(xmb,bmb,geom.x).reshape(geom.nx,geom.ny)
            smb_regrid = bmb_regrid + tmb_regrid
        else:
            print("Need at least of smb and bmb to be defined!")
            
    bmb_regrid = -bmb_regrid #TODO - the negative sign here is very sneaky, make bmb sign consistent and clearer

    return smb_regrid,bmb_regrid

def normalize(geom):
    """
    Normalize the layer thicknesses so that the total thickness matches the thickness of the ice shelf.
    This is necessary to stop the total layer thickness from drifting too far from the ice shelf (drifts are very small for short timescales)

    Arguments:
    geom: Born Geom() object to normalize
    """
    sum_iz = np.sum(geom.d_iso[:,:,geom.layer_mask],axis=2)
    d_iso_out = np.zeros_like(geom.d_iso) #temporary layer thickness
    h_ice = geom.ss-geom.bs
    for iz in geom.layer_mask:
       d_iso_out[:,:,iz] = geom.d_iso[:,:,iz]*h_ice/sum_iz
    geom.d_iso  = d_iso_out
    sum_iz = np.sum(geom.d_iso,axis=2)
    geom.dsum_iso = np.cumsum(geom.d_iso,axis=2)  


def superbee_advection_step(geom,dt):
    """
    Implement the superbee numerical advection scheme on each layer of a Born Geom() object

    Arguments:
    geom: Born Geom() object to advect the layers of
    dt: time step [a]
    """
    fdAdt = np.zeros(shape=(geom.nx,geom.ny,geom.layer_mask.size))

    d_iso = geom.d_iso[:,:,geom.layer_mask].copy()
    d_iso_2l = np.zeros(shape = (geom.nx-2,1,d_iso.shape[2]))
    d_iso_2r = np.zeros(shape = (geom.nx-2,1,d_iso.shape[2]))
    vxsi = np.repeat(geom.vxs2[1:-2, :, np.newaxis], geom.layer_mask.size, axis=2) 
    vxsr = np.repeat(geom.vxs2[2:-1, :, np.newaxis], geom.layer_mask.size, axis=2) 

    d_iso_i = d_iso[1:-1,:,:]
    d_iso_l = d_iso[0:-2,:,:]
    d_iso_r = d_iso[2:,:,:]
    d_iso_2l[1:,:,:] = d_iso[:-3,:,:]
    d_iso_2r[0:-1,:,:] = d_iso[3:,:,:]
    #Flux from the EAST
    thetaE = (vxsr>adv_eps2)*1 + (vxsr<=adv_eps2)*(-1)
    #Construct Slope
    rE = (vxsr<=adv_eps2)*((d_iso_2r-d_iso_r)/(d_iso_r-d_iso_i + adv_eps1)) + (vxsr>adv_eps2)*((d_iso_i - d_iso_l)/(d_iso_r-d_iso_i + adv_eps1))

    #Superbee limiter
    tmp1 = (2*rE<1)*2*rE + (2*rE>=1)*1
    tmp2 = (rE<2)*rE + (rE>=2)*2
    limE = (((tmp1>adv_eps2)*tmp1)>tmp2)*((tmp1>adv_eps2)*tmp1) + (((tmp1>adv_eps2)*tmp1)<=tmp2)*tmp2

    #Flux
    FeV = 0.5 * vxsr *((1+thetaE) * d_iso_i + (1-thetaE)*d_iso_r)
    FeV = FeV + 0.5*np.abs(vxsr)*(1-np.abs(vxsr*dt/geom.dx))*limE*(d_iso_r-d_iso_i)


    #Flux from WEST
    thetaW = (vxsi>adv_eps2)*1 + (vxsi<=adv_eps2)*(-1)
    #Construct Slope
    rW = (vxsi<=adv_eps2)*((d_iso_l-d_iso_2l)/(d_iso_i-d_iso_l + adv_eps1)) + (vxsi>adv_eps2)*((d_iso_r - d_iso_i)/(d_iso_i-d_iso_l + adv_eps1))

    #Superbee limiter
    tmp1 = (2*rW<1)*2*rW + (2*rW>=1)*1
    tmp2 = (rW<2)*rW + (rW>=2)*2
    limW = (((tmp1>adv_eps2)*tmp1)>tmp2)*((tmp1>adv_eps2)*tmp1) + (((tmp1>adv_eps2)*tmp1)<=tmp2)*tmp2

    #Flux
    FwV = 0.5 * vxsi *((1+thetaW) * d_iso_l + (1-thetaW)*d_iso_i)
    FwV = FwV + 0.5*np.abs(vxsi)*(1-np.abs(vxsi*dt/geom.dx))*limW*(d_iso_i-d_iso_l)

    dAdt = -(dt/geom.dx)*(FeV-FwV)
    fdAdt[1:-1,:,:] = dAdt
    
    dQdyst = np.repeat((geom.dQdy/(geom.ss-geom.bs))[:, :, np.newaxis], geom.layer_mask.size, axis=2)
    fdAdt = fdAdt - dQdyst*geom.d_iso[:,:,geom.layer_mask]
    dQdx0 = np.repeat((geom.dQdx/(geom.ss-geom.bs))[:, :, np.newaxis], geom.layer_mask.size, axis=2)[[0,-1],:,:]
    fdAdt[[0,-1],:,:] = fdAdt[[0,-1],:,:] -dQdx0*(geom.d_iso[:,:,geom.layer_mask][[0,-1],:,:])
    geom.d_iso[:,:,geom.layer_mask] = geom.d_iso[:,:,geom.layer_mask]  + fdAdt
    #geom.d_iso[0,:,geom.layer_mask] = geom.d_iso[1,:,geom.layer_mask]
    #geom.d_iso[-1,:,geom.layer_mask] = geom.d_iso[-2,:,geom.layer_mask]
    geom.dsum_iso = np.cumsum(geom.d_iso,axis=2)  


def advection_step(geom,dt):
    """
    Implement a regular numerical advection step (with upwinding) on each layer of a Born Geom() object

    See superbee_advection_step for arguments.
    """
    dAdt  = np.zeros(shape=(geom.nx,geom.ny,geom.layer_mask.size))
    vxst = np.repeat(geom.vxs[:, :, np.newaxis], geom.layer_mask.size, axis=2) 
    dAdt[:-1,:,:] = dAdt[:-1,:,:] - (vxst[1:-1,:]<0)*vxst[1:-1,:,:]*np.diff(geom.d_iso[:,:,geom.layer_mask],1,0)/geom.dx
    dAdt[1:,  :,:] = dAdt[1:,:,:] - (vxst[1:-1,:]>0)*vxst[1:-1,:,:]*np.diff(geom.d_iso[:,:,geom.layer_mask],1,0)/geom.dx
    dAdt[1:-1,:,:] = dAdt[1:-1,:,:]   - geom.d_iso[1:-1,:,geom.layer_mask]*np.diff(vxst[1:-1,:,:],1,0)/geom.dx

    dQdyst = np.repeat((geom.dQdy/(geom.ss-geom.bs))[:, :, np.newaxis], geom.layer_mask.size, axis=2)
    dAdt = dAdt - dQdyst*geom.d_iso[:,:,geom.layer_mask]
    dQdx0 = np.repeat((geom.dQdx/(geom.ss-geom.bs))[:, :, np.newaxis], geom.layer_mask.size, axis=2)[[0,-1],:,:]
    dAdt[[0,-1],:,:] = dAdt[[0,-1],:,:] -dQdx0*(geom.d_iso[:,:,geom.layer_mask][[0,-1],:,:])
    geom.d_iso[:,:,geom.layer_mask] = geom.d_iso[:,:,geom.layer_mask]  + dt*dAdt
    # geom.d_iso[0,:,geom.layer_mask] = geom.d_iso[1,:,geom.layer_mask]
    # geom.d_iso[-1,:,geom.layer_mask] = geom.d_iso[-2,:,geom.layer_mask]
    geom.dsum_iso = np.cumsum(geom.d_iso,axis=2)  


def add_mass_step(geom,smb,bmb):
    """
    Adds/Removes mass from top/bottom layer of a Born Geom() object

    Arguments:
    geom: Born Geom() object
    smb: surface mass balance [m/a]
    bmb: basal mass balance [m/a]
    """
    
    smb_pos = (smb  + np.abs(smb))/2.0
    smb_neg = (smb  - np.abs(smb))/2.0
    geom.d_iso[:,:,geom.top_layer] +=  smb_pos

    #negative surface mass balance:
    for iz in geom.layer_mask[::-1]:
        d_tmp = geom.d_iso[:,:,iz].copy()
        geom.d_iso[:,:,iz] = geom.d_iso[:,:,iz]  + np.maximum(smb_neg, -1.0*d_tmp)
        #remove the same amount from the melting term to keep track how much more needs to be removed:
        smb_neg = smb_neg   + (d_tmp-geom.d_iso[:,:,iz])

    bmb_pos = (bmb  + np.abs(bmb))/2.0
    bmb_neg = (bmb  - np.abs(bmb))/2.0
    geom.d_iso[:,:,geom.bottom_layer] +=  bmb_pos #Refreezing

    for iz in geom.layer_mask:
        d_tmp = geom.d_iso[:,:,iz].copy()
        #remove as much material as possible from the current layer:
        geom.d_iso[:,:,iz] = geom.d_iso[:,:,iz] + np.maximum(bmb_neg, -1.0*d_tmp)
        #remove the same amount from the melting term to keep track how much more needs to be removed:
        bmb_neg= bmb_neg + (d_tmp-geom.d_iso[:,:,iz])

    geom.dsum_iso = np.cumsum(geom.d_iso,axis=2)  

def update_layer_mask(geom):
    """
    Removes 0 layers from layer mask to avoid iterating over them

    Arguments:
    geom: Born Geom() object
    """
    removed_indices = []
    for idx,iz in enumerate(geom.layer_mask):
        if geom.d_iso[:,:,iz].max()<1e-6:
            removed_indices.append(idx)
    temp = np.delete(geom.layer_mask,removed_indices)
    geom.layer_mask = temp.copy()
    del temp


def tracker_step(geom,dt):
    """
    Advects the trackers to follow their respective layers

    Arguments:
    geom: Born Geom() object
    dt: timestep to advect tracker [a]
    """
    for tracker in geom.trackers.trackers:
        if tracker.z>0 +1e-3:
            tracker.update_x(np.minimum(geom.x[-1],tracker.x + geom.vxs[np.argmin(np.abs(geom.x-tracker.x)),0]*dt))
        if tracker.x<geom.x[-1]-1e-3:
            tracker.update_z(geom.dsum_iso[np.argmin(np.abs(geom.x-tracker.x)),0,tracker.layer])

    for pos in geom.tracker_coords:
        tracker_x = np.argmin(np.abs(geom.x-pos[0]))
        tracker_z = np.argmin(np.abs((geom.ss[tracker_x,0]-(geom.dsum_iso[tracker_x,0,:]+geom.bs[tracker_x,0]))-pos[1]))
        tracker_z = geom.top_layer-tracker_z
        geom.trackers.add_tracker(Tracker(geom.x[tracker_x],geom.top_layer-tracker_z,geom.dsum_iso[tracker_x,0,geom.top_layer-tracker_z]))



def sim(geom,smb,bmb,sched:Scheduele):
    """
    Alternative way to simulate the internal stratigraphy of an ice shelf (no buffers which are used in sim)

    See sim() for arguments
    """

    #Here, we add mass to an existing layer that is being advected.
    for n in range (sched.total_iterations):
        sched.update()
        geom.elapsed_t    += sched.dt

        if sched.bool_surface() and sched.bool_base():
            geom.top_layer+=1
            geom.bottom_layer-=1
            geom.layer_mask = np.concatenate(([geom.bottom_layer],geom.layer_mask,[geom.top_layer]))
            add_mass_step(geom,smb*sched.dt,bmb*sched.dt)
            normalize(geom)
        elif sched.bool_surface() and not sched.bool_base():
            geom.top_layer+=1
            geom.layer_mask = np.concatenate((geom.layer_mask,[geom.top_layer]))
            add_mass_step(geom,smb*sched.dt,bmb*sched.dt)
            normalize(geom)
        elif not sched.bool_surface() and sched.bool_base():
            geom.bottom_layer-=1
            geom.layer_mask = np.concatenate(([geom.bottom_layer],geom.layer_mask))
            add_mass_step(geom,smb*sched.dt,bmb*sched.dt)
            normalize(geom)
        else:
            add_mass_step(geom,smb*sched.dt,bmb*sched.dt)

        geom.age_iso[geom.bottom_layer:geom.top_layer+1]+=sched.dt
        if sched.it % 10 == 0:
            update_layer_mask(geom)

        advection_step(geom,sched.dt)
        if sched.it == sched.total_iterations:
            normalize(geom)

        tracker_step(geom,sched.dt)
