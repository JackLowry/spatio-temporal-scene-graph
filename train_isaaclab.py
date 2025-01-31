import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import PIL
from network import StowTrainSceneGraphModel
from data_loader import IsaacLabDataset, StowDataset
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

def randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, node_gt, edge_gt):
    #todo randomize edges

    num_objects = batch_object_bbox.shape[1]
    batch_size = batch_images.shape[0]


    for batch_idx in range(batch_size):
        random_node_idxs = torch.randperm(num_objects)
        batch_object_bbox[batch_idx] = batch_object_bbox[batch_idx, random_node_idxs]
        for i in range(len(node_gt)):
            node_gt[i][batch_idx] = node_gt[i][batch_idx, random_node_idxs]

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
        for i in range(len(edge_gt)):
            edge_gt[i][batch_idx] = edge_gt[i][batch_idx, edge_idxs]

    return batch_images, batch_object_bbox, batch_union_bbox, node_gt, edge_gt

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

config_name = "fast-rcnn.yaml"

def multi_gpu_train(rank, world_size, config):
    ddp_setup(rank, world_size)
    train(rank, config)

def train(device, config):
    wandb.init(project="scenegraphgeneration-stow", group="DDP", config=config)
    generator1 = torch.Generator().manual_seed(42)
    multi_gpu = config["multi_gpu"]

    #process dataset
    root_data_dir = "/home/jack/research/data/isaaclab_sg/01-31-2025:12-58-41"
    root_dir = "/home/jack/research/scene_graph/spatio_temporal_sg"
    preproccess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    scale_factor = 0.25
    dataset = IsaacLabDataset(root_data_dir, scale_factor=scale_factor, transform=preproccess)
    

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1], generator=generator1,)

    if multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)


    training_iterations = 1000

    network_config = config["network_config"]
    model = StowTrainSceneGraphModel(4, dataset.num_object_labels, dataset.num_relationship_labels, **network_config)
    model = model.to(device)

    if multi_gpu:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    if config["use_node_class_weights"]:
        node_label_loss_metric = nn.CrossEntropyLoss(weight=dataset.label_weights)
    else:
        node_label_loss_metric = nn.CrossEntropyLoss()
    edge_label_loss_metric = nn.CrossEntropyLoss()

    pos_loss_metric= nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=3e-4)

    save_interval = 1000
    report_interval = 10
    eval_interval = 50


    step = 0

    for epoch in tqdm(range(2, training_iterations)):
        losses = []
        node_losses = []
        edge_losses = []
        pos_losses = []
        if multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
        

        for batch in (pbar := tqdm(train_dataloader, leave=False)):
        # for batch in train_dataloader:
            optimizer.zero_grad()

            batch_rgb = batch["image"].to(device).to(torch.float)
            batch_images = batch_rgb
            batch_object_bbox = batch["nodes"]["bbox"].to(device)
            batch_union_bbox = batch["edges"]["bbox"].to(device)
            gt_node_label = batch["nodes"]["object_label"].to(device)
            gt_edge_label = batch["edges"]["relationship_label"].to(device)
            gt_pos = batch["edges"]["dist"].to(device)

            # import pdb; pdb.set_trace
            # (batch_images, batch_object_bbox, batch_union_bbox, node_parameters, edge_parameters) \
            #     = randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, [gt_node_label], [gt_edge_label, gt_pos])
            # gt_node_label = node_parameters[0]
            # (gt_edge_label, gt_pos) = edge_parameters
            # not_visible = gt_visible == 0
            # gt_pos[not_visible] = torch.zeros((3)).to(device)

            node_labels, edge_labels, relative_position = model(batch_images, batch_object_bbox, batch_union_bbox)

            node_loss = node_label_loss_metric(node_labels.permute(0, 2, 1), gt_node_label.long())
            edge_loss = edge_label_loss_metric(edge_labels.permute(0, 2, 1), gt_edge_label.long())
            pos_loss = pos_loss_metric(relative_position, gt_pos)

            loss = node_loss + edge_loss + pos_loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            node_losses.append(node_loss.item())
            edge_losses.append(edge_loss.item())
            pos_losses.append(pos_loss.item())
            # pos_losses.append(0)

            pbar.set_description(f"Loss: {loss}")


            if step % report_interval == 0:
                # import pdb; pdb.set_trace()
                if step % eval_interval == 0:
                    with torch.no_grad():
                        test_node_loss = []
                        test_edge_loss = []
                        test_dist_loss = []
                        test_loss = []

                        num_correct_node = 0
                        num_correct_edge = 0
                        num_total_node = 0
                        num_total_edge = 0
                        for test_batch in (pbar := tqdm(test_dataloader, leave=False)):

                            batch_rgb = batch["image"].to(device).to(torch.float)
                            batch_images = batch_rgb
                            batch_object_bbox = batch["nodes"]["bbox"].to(device)
                            batch_union_bbox = batch["edges"]["bbox"].to(device)
                            gt_node_label = batch["nodes"]["object_label"].to(device)
                            gt_edge_label = batch["edges"]["relationship_label"].to(device)
                            gt_pos = batch["edges"]["dist"].to(device)

                            node_labels, edge_labels, relative_position = model(batch_images, batch_object_bbox, batch_union_bbox)
                            node_loss = node_label_loss_metric(node_labels.permute(0, 2, 1), gt_node_label.long())
                            edge_loss = edge_label_loss_metric(edge_labels.permute(0, 2, 1), gt_edge_label.long())
                            pos_loss = pos_loss_metric(relative_position, gt_pos)

                            pred_node_labels = torch.argmax(nn.functional.softmax(node_labels, dim=-1), dim=-1)
                            pred_edge_labels = torch.argmax(nn.functional.softmax(edge_labels, dim=-1), dim=-1)

                            num_correct_node += torch.sum(pred_node_labels == gt_node_label)
                            num_total_node += gt_node_label.shape[0]*gt_node_label.shape[1]
                            num_correct_edge += torch.sum(pred_edge_labels == gt_edge_label)
                            num_total_edge += gt_edge_label.shape[0]*gt_edge_label.shape[1]

                            # import pdb; pdb.set_trace()

                            test_node_loss.append(node_loss)
                            test_edge_loss.append(edge_loss)
                            test_dist_loss.append(pos_loss)
                            test_loss.append(pos_loss)
                        avg_test_loss = sum(test_loss)/len(test_loss)
                        avg_test_node_loss = sum(test_node_loss)/len(test_node_loss)
                        avg_test_edge_loss = sum(test_edge_loss)/len(test_edge_loss)
                        avg_test_pos_loss = sum(test_dist_loss)/len(test_dist_loss)
                        node_accuracy = num_correct_node/num_total_node
                        node_accuracy = node_accuracy.item()
                        edge_accuracy = num_correct_edge/num_total_edge
                        edge_accuracy = edge_accuracy.item()


                avg_loss = sum(losses)/len(losses)
                avg_node_loss = sum(node_losses)/len(node_losses)
                avg_edge_loss = sum(edge_losses)/len(edge_losses)
                avg_pos_loss = sum(pos_losses)/len(pos_losses)
                    
                # viz_labels = torch.concat((visible_pred[0], gt_visible[0]), dim=1)
                # viz_pos_losses = torch.sqrt(torch.sum(torch.square(relative_position[0] - gt_pos[0]), dim=1))
                # fig = visualize_boxes(batch_images[0], batch_object_bbox[0], batch_union_bbox[0], viz_labels, viz_pos_losses)
                # rgb_image = wandb.Image(batch_rgb[0].cpu().permute(1,2,0).to(torch.int).numpy())
                # seg_image = batch_seg[0].cpu().permute(1,2,0).to(torch.int).numpy()
                # seg_image = seg_image/np.max(seg_image)*255
                # seg_image = wandb.Image(seg_image)
                # node_viz_image = wandb.Image("nodes.png")
                # edge_viz_image = wandb.Image("edges.png")

                node_labels_viz = node_labels[0]
                node_labels_viz = torch.argmax(nn.functional.softmax(node_labels_viz, dim=-1), dim=-1)
                img_language_labels = [dataset.metadata["object_id_to_name"][idx.cpu().item()] for idx in node_labels_viz]
                img = draw_image(batch["orig_image"][0], batch_object_bbox.cpu()[0]*(1/scale_factor), img_language_labels)
                img.save("test.png")
                img = wandb.Image(img)
                if (not multi_gpu or device == 0) and step % eval_interval != 0:
                    wandb.log({"train/loss": avg_loss, "train/node_loss": avg_node_loss, "train/edge_loss": avg_edge_loss, "train/pos_loss": pos_loss, "img": img})
                else:
                    wandb.log({"test/node_accuracy": node_accuracy, "test/edge_accuracy": edge_accuracy,
                            "train/loss": avg_loss, "train/node_loss": avg_node_loss, "train/edge_loss": avg_edge_loss, "train/pos_loss": pos_loss, 
                            "test/loss": avg_test_loss, "test/node_loss": avg_test_node_loss, "test/edge_loss": avg_test_edge_loss, 
                            "test/pos_loss": avg_test_pos_loss, "img": img})
                    
            if (multi_gpu and device == 0) and step % save_interval == 0:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                torch.save(model.module.state_dict(), f"ckpts/loss_{avg_loss}_{timestr}.pt")
            elif not multi_gpu and step % save_interval == 0:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                torch.save(model.state_dict(), os.path.join(root_dir, f"ckpts/loss_{avg_loss}_{timestr}.pt"))
            step += 1

    if multi_gpu:
        destroy_process_group() 
    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config)

    if config["multi_gpu"]:
        world_size = torch.cuda.device_count()
        mp.spawn(multi_gpu_train, args=(world_size, config,), nprocs=world_size)

    else:
        device = torch.device("cuda:0")
        device = torch.device("cpu")
        train(device, config)
        

if __name__ == "__main__":
    main()




        

