#!/bin/bash

declare -a arr=("fast-rcnn")

for i in "${arr[@]}"
do
    sbatch train_stow.slurm "$i"
done
