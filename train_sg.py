import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import PIL
from network import FullSceneGraphModel
from data_loader import SceneGraphDataset
from visualization import visualize_boxes

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import time

import wandb

def randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, gt_visible, gt_pos):
    #todo randomize edges

    num_objects = batch_object_bbox.shape[1]
    batch_size = batch_images.shape[0]


    for batch_idx in range(batch_size):
        random_node_idxs = torch.randperm(num_objects)
        batch_object_bbox[batch_idx] = batch_object_bbox[batch_idx, random_node_idxs]
        gt_visible[batch_idx] = gt_visible[batch_idx, random_node_idxs]

        num_edges = num_objects*num_objects
        edge_idxs = torch.arange(num_edges)

        non_center_edges = edge_idxs % (num_objects+1) != 0

        edge_idxs = torch.stack(torch.split(edge_idxs, num_objects), dim=0)
        edge_idxs = edge_idxs[random_node_idxs]
        edge_idxs = [e[random_node_idxs] for e in edge_idxs]
        edge_idxs = torch.concat(edge_idxs)
        edge_idxs = edge_idxs[non_center_edges]
        edge_idxs = edge_idxs - edge_idxs//num_objects

        batch_union_bbox[batch_idx] = batch_union_bbox[batch_idx, edge_idxs]
        gt_pos[batch_idx] = gt_pos[batch_idx, edge_idxs]

    return batch_images, batch_object_bbox, batch_union_bbox, gt_visible, gt_pos


wandb.init(project="scenegraphgeneration")

root_data_dir = "/mmfs1/home/jrl712/amazon_home/data/sg_data_gt"
dataset = SceneGraphDataset(root_data_dir)
train_dataloader = DataLoader(dataset, batch_size=64a, shuffle=True)

device = torch.device("cuda:0")

training_iterations = 1000

model = FullSceneGraphModel(10, dataset.).to(device)

visible_loss_metric = nn.CrossEntropyLoss()
pos_loss_metric= nn.MSELoss()

optimizer = Adam(model.parameters(), lr=0.0001)

save_interval = 5

for epoch in tqdm(range(2, training_iterations)):
    losses = []
    visible_losses = []
    pos_losses = []
    for batch in (pbar := tqdm(train_dataloader, leave=False)):
        optimizer.zero_grad()

        batch_rgb = batch["images"]["rgb"].to(device).to(torch.float)
        batch_seg = batch["images"]["seg"].to(device).to(torch.float).unsqueeze(1)
        batch_depth = batch["images"]["depth"].to(device).to(torch.float).unsqueeze(1)


        # batch_images = torch.concat((batch_rgb, batch_seg, batch_depth), dim=1)
        batch_images = batch_rgb

        batch_object_bbox = batch["nodes"]["bbox"].to(device)
        batch_union_bbox = batch["edges"]["bbox"].to(device)
        gt_visible = batch["nodes"]["visible"].to(device)
        gt_visible = nn.functional.one_hot(gt_visible.to(torch.long)).to(torch.float)
        gt_pos = batch["edges"]["relative_pos"].to(device)


        (batch_images, batch_object_bbox, batch_union_bbox, gt_visible, gt_pos) = randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, gt_visible, gt_pos)

        # not_visible = gt_visible == 0
        # gt_pos[not_visible] = torch.zeros((3)).to(device)

        visible_pred, pos_pred = model(batch_images, batch_object_bbox, batch_union_bbox)

        visible_loss = visible_loss_metric(visible_pred, gt_visible)
        pos_loss = 10*pos_loss_metric(pos_pred, gt_pos)

        loss = visible_loss + pos_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        visible_losses.append(visible_loss.item())
        pos_losses.append(pos_loss.item())
        # pos_losses.append(0)

        pbar.set_description(f"Loss: {loss}")
    
    avg_loss = sum(losses)/len(losses)
    avg_visible_loss = sum(visible_losses)/len(visible_losses)
    avg_pos_loss = sum(pos_losses)/len(pos_losses)

    if epoch % save_interval == 0:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        torch.save(model.state_dict, f"ckpts/loss_{avg_loss}_{timestr}.pt")
        
    viz_labels = torch.concat((visible_pred[0], gt_visible[0]), dim=1)
    viz_pos_losses = torch.sqrt(torch.sum(torch.square(pos_pred[0] - gt_pos[0]), dim=1))
    fig = visualize_boxes(batch_images[0], batch_object_bbox[0], batch_union_bbox[0], viz_labels, viz_pos_losses)
    rgb_image = wandb.Image(batch_rgb[0].cpu().permute(1,2,0).to(torch.int).numpy())
    seg_image = batch_seg[0].cpu().permute(1,2,0).to(torch.int).numpy()
    seg_image = seg_image/np.max(seg_image)*255
    seg_image = wandb.Image(seg_image)
    node_viz_image = wandb.Image("nodes.png")
    edge_viz_image = wandb.Image("edges.png")
    wandb.log({"loss": avg_loss, "visible_loss": visible_loss, "pos_loss": pos_loss, 'rgb': rgb_image, "node_viz": node_viz_image, "edge_viz": edge_viz_image,"seg": seg_image})






        

