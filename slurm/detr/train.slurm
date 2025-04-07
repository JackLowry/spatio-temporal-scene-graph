#!/bin/bash

#SBATCH --job-name=pretrain_detr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrl712@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=512G
#SBATCH --gpus=a40:8
#SBATCH --time=00-48:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/jrl712
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/detr/logs/stdout.txt # where STDOUT goes
#SBATCH --error=/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/detr/logs/stderr.txt # where STDERR goes

# Modules to use (optional).

# Your programs to run.
/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/detr/train.sh $1
