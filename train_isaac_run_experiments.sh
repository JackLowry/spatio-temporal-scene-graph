#!/bin/bash

declare -a arr=("networks/imp-iterations/one" "networks/imp-iterations/two" "networks/imp-iterations/three")

for i in "${arr[@]}"
do
    sbatch train_isaac.slurm "$i"
done
