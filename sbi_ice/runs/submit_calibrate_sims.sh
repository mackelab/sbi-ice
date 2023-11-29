#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-01:00            # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=16G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL

#scontrol show job $SLURM_JOB_ID

workflow(){
    eval "$(conda shell.bash hook)"
    conda activate sbi_ice
    cd $0
    python calibrate_sims.py
    conda deactivate
}

workflow