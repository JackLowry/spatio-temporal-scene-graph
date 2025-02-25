#!/bin/bash

declare -a arr=("networks/imp-iterations/two" "networks/temporal/two")

for i in "${arr[@]}"
do
    sbatch train.slurm "$i"
done
