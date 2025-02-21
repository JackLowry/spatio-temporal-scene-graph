#!/bin/bash

declare -a arr=("networks/imp-iterations/one" "networks/imp-iterations/two" "networks/imp-iterations/three" "networks/temporal/two")

for i in "${arr[@]}"
do
    sbatch train.slurm "$i"
done
