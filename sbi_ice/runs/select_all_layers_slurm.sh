#!/bin/bash
export shelf=Ekstrom
export exp=exp2
export gt_version=v0
export selection_method=advanced_noise

for job in {3..199..1}; do
    export job=${job}
    sbatch -p cpu-preemptable submit_select_layers.sh  $0
done