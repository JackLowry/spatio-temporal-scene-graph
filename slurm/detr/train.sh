#!/bin/bash

cd /mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph
source ~/.bashrc
conda run -n egtr --live-stream python3 pretrain_detr.py -cn $1
