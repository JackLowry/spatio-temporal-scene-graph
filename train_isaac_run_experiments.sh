#!/bin/bash

declare -a arr=("networks/fast-rcnn", "networks/fast-rcnn-lstm")

for i in "${arr[@]}"
do
    sbatch train_isaac.slurm "$i"
done
