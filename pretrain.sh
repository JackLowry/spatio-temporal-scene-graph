#!/bin/bash

cd /mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph
source ~/.bashrc
conda run -n scenegraphgen2 --live-stream python3 pretrain.py -cn $1
