#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-00:10            # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=hostname_submit_select_layers.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=hostname_submit_select_layers.err   # File to which STDERR will be written - make sure this is not on $HOME


#scontrol show job $SLURM_JOB_ID

workflow(){
    eval "$(conda shell.bash hook)"
    conda activate sbi_ice
    cd $0
    python select_layers.py $shelf $exp $gt_version $selection_method $job
}

workflow