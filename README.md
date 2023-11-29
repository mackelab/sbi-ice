# Simulation-Based Inference of Surface Accumulation and Basal Melt Rates of Antarctic Ice Shelves from Isochronal Layers

This repository contains research code for Simulation-based inference of surface accumulation and basal melt rates of Antarctic Ice Shelves from Isochronal Layers [preprint link]. It contains scripts for the forward model, along with the inference workflow. For code used to preprocess ice shelf data, and generate synthetic ice shelves using [firedrake](https://www.firedrakeproject.org) and [icepack](https://icepack.github.io) - see [preprocessing_ice_data](https://github.com/mackelab/preprocessing_ice_data/). 
Maintenance is ongoing! Expect improvements in terms of extended functionality and improved usability in the future.

Some small data and output files are already included in this repository for quicker start-up. Some parts of the workflow require bigger files. These will clearly be marked in the **workflow** section, along with links to where these files can be downloaded.
The picked IRH data for Ekström Ice Shelf can be found [here](https://doi.org/doi:10.5281/zenodo.10231043).

## Installation
Activate a virtual environment, e.g. using conda. Install dependencies:
```
    git clone https://github.com/mackelab/sbi_ice.git
    cd sbi_ice
    pip install -e .
```

## Workflow

### TL;DR
The full workflow (from running the simulator to evaluating posterior preditictive distributions) involves the following steps:
1. Simulate many times using `sbi_ice/runs/run_sims.py`. This can be parallelized in hydra by running `python run_sims.py -m`.
2. Allocate some simulations as calibration simulations by moving them under `out/$shelf$/exp/calib_sims`
    - `sbi_ice/runs/calibrate_sims.py` to find anomalies and layer bounds on `calib_sims` (make sure correct folder is used in `calibrate_sims.py`). If noise model needs fitting. Also set `noise_dist_overwrite = True`.
    - `sbi_ice/runs/calibrate_sims.py` to find anomalies in `layer_sims` (check correct folder is used in `calibrate_sims.py` and that `layer_bounds_overwrite`, `noise_dist_overwrite` are set to `False`). This can be done in parallel across many jobs using `sbi_ice/runs/submit_calibrate_sims.sh`
3. Add noise to all layers and select best fitting layer for each simulation in each job, using `sbi_ice/runs/select_layers.py`(can be done in parallel using `sbi_ice/runs/submit_select_layers_local.sh`). If simulations done across many jobs, concatenate all outputs of step 3. using `notebooks/workflow/combine_pickles.ipynb`
4. Train a Neural Posterior Estimation (NPE) density estimator using `sbi_ice/runs/train_NPE.py`. This can be parallelized in hydra by running `python train_NPE.py -m`.
5. Perform posterior predictive simulations using `sbi_ice/runs/post_predictive.py`. This can be parallelized in hydra by running `python post_predictive.py -m`.




### 1. Running the simulator

The layer tracing simulator is defined under `sbi_ice.simulators.Layer_Tracing_Sim.py`.
A .csv file defining a flowline domain and discretized variables is required to define the simulator. Some suitable files are found under `data/$shelf/setup_files`.
A tutorial for how the simulator is used is found under `notebooks/tutorials/layer_tracer_tutorial.ipynb`

We can submit jobs to perform many simulations at once using `sbi_ice/runs/run_sims`. Configuration files describing the simulation length,resolution, prior distribution are found under `configs/run_sims`. The simulations are saved in separate per-job folders in `out/$shelf/$exp/$layer_sims`

### 2. Calibrating the simulations

At this step we assume to have a set of simulations performed in jobs. The output job folders should be stored under `out/$shelf/$exp/layer_sims`. We calibrate some quantities before performing inference. Therefore we require a distinct set of calibration sims. We move one of the jobs from `./layer_sims` to `./calib_sims`.

First, we look at the calibration simulations. We find (and ignore) any anomalous simulations that might have occured. We then find the LMI boundary as described in the preprint. This gives us the mask we should apply on the layer elevation data that is dependent on the arbitrary boundary condition. Finally, if we need to calibrate the noise model (as is the case for Ekström Ice Shelf), we do this here. This is done using `sbi_ice/runs/calibrate_sims.py`. In the file, make sure the following parameters at the top are set correctly:
```
    shelf
    exp
    gt_version
    anomalies_overwrite = True
    layer_bounds_overwrite = True
    noise_dist_overwrite = True/False
    sim_or_calib = "layer_sims"/"calib_sims"
```

We then find any anomalous results in `.layer_sims`. Keep everything the same, and make sure `layer_bounds_overwrite=False` and `noise_dist_overwite = False`. This can be sped up given access to multiple CPU cores by running `sbatch submit_calibrate_sims.sh` on SLURM or `source submit_calibrate_sims.sh` locally.

### 3. Noise model and best fitting layer selection

Our inference workflow requires only one layer out of each simulation to be the observed value, and so we select the best fitting layer (by MSE). This is done for all simulations in a single job using the script `sbi_ice/runs/select_layers.py`. This can be parallelized across all the simulation jobs using `sbi_ice_runs/submit_select_layers_local.sh` or `sbi_ice/runs/submit_select_layers_slurm.sh`
The rest of the workflow takes in one processed layer file for all simulations - the per-job files can be combined and saved into one file using `notebooks/workflow/combine_pickles.ipynb`

To reproduce the results reported in the paper, you can [download the posteriors and processed simulation data](https://doi.org/doi:10.5281/zenodo.10245153), and save the respective files in `out`.


### 4. Training a Density Estimator

The script `sbi_ice/runs/train_NPE.py` is used to train a density estimator to estimate the posterior distribution for a given experiment for one of the real layers in that experiment. Configurations such as which experiment to use, the network architecture, and which posterior to learn are found under `configs/training`. The script is run with `hydra` using `python train_NPE.py -m` to submit several jobs for different layer numbers and/or different random seeds.

The trained posteriors for our experiments can be found in [link to data repository]. However, they need to be saved at the correct locations, with the Ekstrom example being saved under `out/Ekstrom/exp2/sbi_sims/posteriors`, and the synthetic example under `out/Synthetic_long/exp3/sbi_sims/posteriors`

### 5. Evaluating Posterior Predictive Distributions

Given a trained posterior density estimator, we can sample from this posterior and simulate using these samples with the script `sbi_ice/runs/post_predictive.py`. Configurations are found under `configs/training`. The script is run with `hydra` using `python train_NPE.py -m` to submit several jobs for different layer numbers and/or different random seeds.