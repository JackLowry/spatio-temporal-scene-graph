#!/bin/bash

declare -a arr=("fast-rcnn")

for i in "${arr[@]}"
do
    sbatch pretrain.slurm "$i"
done
