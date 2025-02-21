#!/bin/bash

#SBATCH --job-name=train_temporal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrl712@uw.edu

#SBATCH --account=sciencehub
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=a40:1
#SBATCH --time=00-48:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/home/jrl712
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/temporal_training/logs/stdout.txt # where STDOUT goes
#SBATCH --error=/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/temporal_training/logs/stderr.txt # where STDERR goes

# Modules to use (optional).

# Your programs to run.
apptainer exec --nv --overlay /gscratch/sciencehub/jrl712/conda_overlay3.img:ro /gscratch/sciencehub/jrl712/hyak-container.sif /mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph/slurm/temporal_training/train.sh $1
