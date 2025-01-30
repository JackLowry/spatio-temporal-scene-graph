import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import PIL
from network import StowTrainSceneGraphModel
from data_loader import StowDataset
from visualization import visualize_boxes, draw_image
import sys

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import time

import wandb    

from omegaconf import DictConfig, OmegaConf
import hydra

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from paramiko import SSHClient
from scp import SCPClient
import json

config_name = "fast-rcnn.yaml"

def eval(device, config):
    generator1 = torch.Generator().manual_seed(42)
    multi_gpu = config["multi_gpu"]


    #process dataset
    root_data_dir = "/mmfs1/home/jrl712/amazon_home/data/bin_syn"
    preproccess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dataset = StowDataset(root_data_dir, 5, scale_factor=1, transform=preproccess)
    

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1], generator=generator1,)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)



    network_config = config["network_config"]
    model = StowTrainSceneGraphModel(10, dataset.num_object_labels, dataset.num_relationship_labels, **network_config)
    model.load_state_dict(torch.load("ckpts/loss_0.2939637872305783_2024_10_30-01_46_07.pt", weights_only=True))
    model = model.to(device)


        

    for batch in (pbar := tqdm(train_dataloader, leave=False)):
    # for batch in train_dataloader:

        batch_rgb = batch["image"].to(device).to(torch.float)
        batch_images = batch_rgb
        batch_object_bbox = batch["nodes"]["bbox"].to(device)
        batch_union_bbox = batch["edges"]["bbox"].to(device)
        gt_node_label = batch["nodes"]["object_label"].to(device)
        gt_edge_label = batch["edges"]["relationship_label"].to(device)
        gt_pos = batch["edges"]["dist"].to(device)

        node_labels, edge_labels, relative_position = model(batch_images, batch_object_bbox, batch_union_bbox)


        node_labels_viz = node_labels[0]
        node_labels_viz = torch.argmax(nn.functional.softmax(node_labels_viz, dim=-1), dim=-1)
        edge_labels_viz = edge_labels[0]
        edge_labels_viz = torch.argmax(nn.functional.softmax(edge_labels_viz, dim=-1), dim=-1)
        
        img_language_labels = [dataset.scene_graph_metadata["object_id_to_name"][str(idx.cpu().item())] for idx in node_labels_viz]
        img = draw_image(batch["orig_image"][0], batch_object_bbox.cpu()[0], img_language_labels)
        img.save("sg.png")


        objects = []
        for o_id in node_labels_viz:
            objects.append({'name': dataset.scene_graph_metadata["object_id_to_name"][str(o_id.cpu().item())]})

        relationships = []
        for idx, r_id in enumerate(edge_labels_viz):
            subject_idx = batch["edges"]["idx"][idx] // dataset.num_objects
            object_idx = batch["edges"]["idx"][idx] % dataset.num_objects
            relationships.append({
                'predicate': dataset.scene_graph_metadata["edge_id_to_name"][str(r_id.cpu().item())],
                'subject': subject_idx,
                'object': object_idx,
            })
        
        graphviz_sg = {
            'url': "sg.png",
            'attributes': [],
            'objects': objects,
            'relationships': relationships
        }
        f = open('scene_graph.js', 'w')
        f.write('var graph = ' + json.dumps(graphviz_sg))
        f.close()

        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect('bicycle.cs.washington.edu', username='jrl712')
        scp = SCPClient(ssh.get_transport())
        scp.put('scene_graph.js', remote_path='/cse/web/homes/jrl712/GraphViz')
        scp.put('sg.png', remote_path='/cse/web/homes/jrl712/GraphViz')
        break

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config)
    device = torch.device("cuda:0")
    eval(device, config)
        

if __name__ == "__main__":
    main()




        

