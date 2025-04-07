#!/bin/bash

declare -a arr=("default")

for i in "${arr[@]}"
do
    sbatch train.slurm "$i"
done
