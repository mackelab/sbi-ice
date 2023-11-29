#!/bin/bash
export shelf=Synthetic_long
export exp=exp3
export gt_version=v0
export selection_method=advanced_noise

for job in {1..4..1}; do
    export job=${job}
    source submit_select_layers.sh $0
done