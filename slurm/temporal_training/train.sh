#!/bin/bash

cd /mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph
source ~/.bashrc
conda run -n scenegraphgen --live-stream python3 train_temporal.py -cn $1
