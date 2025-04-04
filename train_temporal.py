from loss import ContrastiveLoss
import metrics
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import PIL
from network import StowTrainSceneGraphModel
from network_temporal import TemporalSceneGraphModel
from data_loader import IsaacLabDataset, IsaacLabTemporalDataset, StowDataset
from visualization import visualize_boxes, draw_image, draw_graph
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

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs,
                             node_gt, edge_gt):
    #todo randomize edges
    num_objects = batch_object_bbox.shape[2]
    batch_size = batch_images.shape[0]
    seq_len = batch_images.shape[1]
    
    batch_images = batch_images.flatten(0, 1)
    batch_object_bbox = batch_object_bbox.flatten(0, 1) 
    batch_union_bbox = batch_union_bbox.flatten(0, 1)
    batch_edge_idx_to_node_idxs = batch_edge_idx_to_node_idxs.flatten(0, 1)

    for i in range(len(node_gt)):
        node_gt[i] = node_gt[i].flatten(0, 1)

    for i in range(len(edge_gt)):
        edge_gt[i] = edge_gt[i].flatten(0, 1)

    for batch_idx in range(batch_size*seq_len):
        random_node_idxs = torch.randperm(num_objects).to(batch_images.device)
        batch_object_bbox[batch_idx] = batch_object_bbox[batch_idx, random_node_idxs]
        for i in range(len(node_gt)):
            node_gt[i][batch_idx] = node_gt[i][batch_idx, random_node_idxs]

        num_edges = num_objects*num_objects
        edge_idxs = torch.arange(num_edges).to(batch_images.device)

        non_self_edges = edge_idxs % (num_objects+1) != 0

        edge_idxs = torch.stack(torch.split(edge_idxs, num_objects), dim=0)
        edge_idxs = edge_idxs[random_node_idxs]
        edge_idxs = [e[random_node_idxs] for e in edge_idxs]
        edge_idxs = torch.concat(edge_idxs)

        non_self_edges = torch.stack(torch.split(non_self_edges, num_objects), dim=0)
        non_self_edges = non_self_edges[random_node_idxs]
        non_self_edges = [e[random_node_idxs] for e in non_self_edges]
        non_self_edges = torch.concat(non_self_edges)

        #remove self edges
        edge_idxs = edge_idxs[non_self_edges]

        edge_idxs = edge_idxs - (1+edge_idxs//(num_objects+1))

        batch_edge_idx_to_node_idxs[batch_idx] = batch_edge_idx_to_node_idxs[batch_idx][edge_idxs]
        batch_edge_idx_to_node_idxs[batch_idx][:, 1] = random_node_idxs[batch_edge_idx_to_node_idxs[batch_idx][:, 1].to(torch.long)]
        batch_edge_idx_to_node_idxs[batch_idx][:, 2] = random_node_idxs[batch_edge_idx_to_node_idxs[batch_idx][:, 2].to(torch.long)]


        batch_union_bbox[batch_idx] = batch_union_bbox[batch_idx, edge_idxs]
        for i in range(len(edge_gt)):
            edge_gt[i][batch_idx] = edge_gt[i][batch_idx, edge_idxs]

    batch_images = batch_images.reshape(batch_size, seq_len, *batch_images.shape[1:])
    batch_object_bbox = batch_object_bbox.reshape(batch_size, seq_len, *batch_object_bbox.shape[1:])
    batch_union_bbox = batch_union_bbox.reshape(batch_size, seq_len, *batch_union_bbox.shape[1:])
    batch_edge_idx_to_node_idxs = batch_edge_idx_to_node_idxs.reshape(batch_size, seq_len, *batch_edge_idx_to_node_idxs.shape[1:])

    for i in range(len(node_gt)):
        node_gt[i] = node_gt[i].reshape(batch_size, seq_len, *node_gt[i].shape[1:])

    for i in range(len(edge_gt)):
        edge_gt[i] = edge_gt[i].reshape(batch_size, seq_len, *edge_gt[i].shape[1:])

    return batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs, node_gt, edge_gt

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

config_name = "networks/temporal/zero.yaml"

# def multi_gpu_train(rank, world_size, config):
#     ddp_setup(rank, world_size)
#     train(rank, config)

class Trainer():
    
        def __init__(self, device, config):
            self.device = device
            self.config = config

            self.contrastive_loss_fn = ContrastiveLoss()

        def batch_forward(self, model, batch):
            device = self.device
            config = self.config
            batch_rgb = batch["image"].to(device).to(torch.float).clone()
            batch_images = batch_rgb
            batch_object_bbox = batch["nodes"]["bbox"].to(device).clone()
            batch_union_bbox = batch["edges"]["bbox"].to(device).clone()
            batch_edge_idx_to_node_idxs = batch["edge_idx_to_node_idxs"].to(device).clone()
            gt_node_label = batch["nodes"]["object_label"].to(device).clone()
            gt_edge_label = batch["edges"]["relationship_label"].to(device).clone()
            gt_pos = batch["edges"]["dist"].to(device).clone()

            batch_node_network_mask = batch['node_network_mask'].to(device).clone()
            batch_edge_network_mask = batch['edge_network_mask'].to(device).clone()

            if config["randomize_input"] and config["randomize_ground_truth"]:
                (batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs, node_parameters, edge_parameters)   \
                    = randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs,
                                            [gt_node_label, batch_node_network_mask], [gt_edge_label, gt_pos, batch_edge_network_mask])
                (gt_node_label, batch_node_network_mask) = node_parameters
                (gt_edge_label, gt_pos, batch_edge_network_mask) = edge_parameters
            elif config["randomize_input"]:
                (batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs, node_parameters, edge_parameters) \
                = randomize_graph_ordering(batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs,
                                            [batch_node_network_mask], [batch_edge_network_mask])
                batch_node_network_mask = node_parameters[0]
                batch_edge_network_mask = edge_parameters[0]

            model_outs = model(batch_images, 
                                                                    batch_object_bbox, 
                                                                    batch_union_bbox, 
                                                                    batch_edge_idx_to_node_idxs,
                                                                    batch_node_network_mask,
                                                                    batch_edge_network_mask)
            
            return model_outs

            # node_labels = node_labels.reshape(-1, node_labels.shape[-1])
            # gt_node_label = gt_node_label.reshape(-1)
            # batch_node_network_mask = batch_node_network_mask.reshape(-1)

            # edge_labels = edge_labels.reshape(-1, edge_labels.shape[-1])
            # gt_edge_label = gt_edge_label.reshape(-1)
            # batch_edge_network_mask = batch_edge_network_mask.reshape(-1)


            # relative_position = relative_position.reshape(-1, relative_position.shape[-1])
            # gt_pos = gt_pos.reshape(-1, gt_pos.shape[-1])

        def get_losses(self, model_outs, batch):
            device = self.device
            config = self.config
            gt_node_label = batch["nodes"]["object_label"].to(device).clone()
            gt_node_bbox = batch["nodes"]["bbox"].to(device).clone()
            # node_loss = node_label_loss_metric(node_labels[batch_node_network_mask], gt_node_label.long()[batch_node_network_mask])
            # edge_loss = edge_label_loss_metric(edge_labels[batch_edge_network_mask], gt_edge_label.long()[batch_edge_network_mask])
            # edge_loss[torch.isnan(edge_loss)] = 0.0
            # pos_loss = pos_loss_metric(relative_position[batch_edge_network_mask], gt_pos[batch_edge_network_mask])
            # pos_loss[torch.isnan(pos_loss)] = 0.0

            node_latents, pred_node_box = model_outs

            pred_node_box = pred_node_box.reshape(-1, 4)
            gt_node_bbox = gt_node_bbox.reshape(-1, 4)
            pred_node_box = box_cxcywh_to_xyxy(pred_node_box) 
            pred_node_box[:, [0, 2]] *= self.dataset.desired_size[0]
            pred_node_box[:, [1, 3]] *= self.dataset.desired_size[1]

            node_giou_loss = 1 - torch.diag(
                                                generalized_box_iou(
                                                    pred_node_box, 
                                                    gt_node_bbox
                                                    )
                                                )
            node_giou_loss = node_giou_loss.sum()/pred_node_box.shape[0]

            node_l1_box_loss = torch.nn.functional.l1_loss(pred_node_box, gt_node_bbox, reduction='mean')

            #flatten the sequence down. each item is a unique label for an object in the scene.
            node_labels_flat = torch.flatten(gt_node_label, 1, 2)
            node_latent_flat = torch.flatten(node_latents, 1, 2)

            contrastive_loss = 0

            num_nodes = node_labels_flat.shape[-1]
            batch_size = node_labels_flat.shape[0]
            for batch_idx in range(batch_size):
                for node_idx in range(num_nodes):
                    node_label = node_labels_flat[batch_idx, node_idx]
                    if node_label == self.dataset.no_object_label:
                        continue

                    other_idxs = [i for i in range(num_nodes) if i != node_idx]
                    other_labels = node_labels_flat[batch_idx, other_idxs]

                    curr_latent = node_latent_flat[batch_idx, node_idx]
                    other_latents = node_latent_flat[batch_idx, other_idxs]

                    matching_label = other_labels == node_label

                    contrastive_loss += self.contrastive_loss_fn.get_loss(
                        curr_latent, 
                        other_latents,
                        matching_label
                    )

            loss_dict = {
                "node_box_giou": node_giou_loss,
                "node_box_l1": node_l1_box_loss,
                "contrastive_loss": contrastive_loss,
            }
            loss = node_giou_loss + node_l1_box_loss + contrastive_loss#node_loss + edge_loss + pos_loss
            return loss, loss_dict

        def train(self):
            self.device = 0
            wandb.init(project="scenegraphgeneration-isaac", group="DDP", config=self.config)
            run_name = f'{wandb.run.name}_{wandb.run.id}'
            generator1 = torch.Generator().manual_seed(42)
            multi_gpu = self.config["multi_gpu"]

            root_data_dir = self.config['root_data_dir']
            root_dir = self.config['root_dir']
            os.mkdir(os.path.join(root_dir, 'ckpts', run_name))
            preproccess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            scale_factor = self.config['scale_factor']
            batch_size = self.config['batch_size']
            dataset = IsaacLabTemporalDataset(root_data_dir, scale_factor=scale_factor, transform=preproccess)
            self.dataset = dataset

            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1], generator=generator1,)

            if multi_gpu:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(test_dataset))
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


            training_iterations = 1000

            network_config = self.config["network_config"]
            model = TemporalSceneGraphModel(batch_size, dataset.num_objects, dataset.num_object_labels, dataset.num_relationship_labels, dataset.num_objects, network_config)
            model = model.to(self.device)

            if multi_gpu:
                model = DDP(model, device_ids=[self.device], find_unused_parameters=True)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 

            if self.config["use_node_class_weights"]:
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
                loss_dicts = {}
                if multi_gpu:
                    train_dataloader.sampler.set_epoch(epoch)
                

                for batch in (pbar := tqdm(train_dataloader, leave=False)):
                # for batch in train_dataloader:
                    batch = batch
                    optimizer.zero_grad()

                    model_outs = self.batch_forward(model, batch)

                    loss, loss_dict = self.get_losses(model_outs, batch)

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    for k,v in loss_dict.items():
                        if k not in loss_dicts.keys():
                            loss_dicts[k] = []
                        loss_dicts[k].append(v.item())

                    pbar.set_description(f"Loss: {loss}")


                    if step % report_interval == 0:
                        # import pdb; pdb.set_trace()
                        if step % eval_interval == 0:
                            with torch.no_grad(): 
                                test_losses = []
                                test_loss_dicts = {
                                    "node_box_giou": [],
                                    "contrastive_loss": [],
                                }
                                for test_batch in (pbar := tqdm(test_dataloader, leave=False)):

                                    model_outs = self.batch_forward(model, batch)
                                    loss, loss_dict = self.get_losses(model_outs, batch)
                                    test_losses.append(loss.item())
                                    for k,v in loss_dict.items():
                                        if k not in test_loss_dicts.keys():
                                            test_loss_dicts[k] = []
                                        test_loss_dicts[k].append(v.item())

                                avg_test_losses = sum(test_losses)/len(test_losses)
                                avg_test_loss_dicts = {}
                                for k,v in test_loss_dicts.items():
                                    avg_test_loss_dicts[k] = sum(v)/len(v)

                        avg_losses = sum(losses)/len(losses)
                        avg_loss_dicts = {}
                        for k,v in loss_dicts.items():
                            avg_loss_dicts[k] = sum(v)/len(v)



                        seq_id_to_viz = 3

                        gt_node_label = batch["nodes"]["object_label"].clone()
                        # gt_edge_label = batch["nodes"]["object_label"]

                        node_latents, pred_node_box = model_outs
                        pred_node_box = pred_node_box[0][seq_id_to_viz].cpu()
                        pred_node_box = box_cxcywh_to_xyxy(pred_node_box) 
                        pred_node_box[:, [0, 2]] *= self.dataset.desired_size[0]
                        pred_node_box[:, [1, 3]] *= self.dataset.desired_size[1]

                        node_labels_viz = gt_node_label.reshape(batch_size, dataset.num_objects, -1)
                        node_labels_viz = node_labels_viz[0, seq_id_to_viz]
                        # edge_labels_viz = gt_edge_label.reshape(batch_size, dataset.num_objects, -1)
                        # edge_labels_viz = edge_labels_viz[0, seq_id_to_viz]
                        img_language_labels = [dataset.metadata["object_id_to_name"][idx.cpu().item()] for idx in node_labels_viz]
                        # edge_language_labels = [dataset.metadata["edge_id_to_name"][idx.cpu().item()] for idx in edge_labels_viz]
                        gt_img = draw_graph(batch["orig_image"][0][seq_id_to_viz], 
                                            pred_node_box*(1/scale_factor), 
                                            img_language_labels,
                                            None)
                        
                        # img.save("test.png")
                        # img = wandb.Image(img)

                        report_dict = {
                            "train/loss": avg_losses,
                            "img": gt_img
                        }

                        for k,v in avg_loss_dicts.items():
                            report_dict[f"train/{k}"] = v

                        if (not multi_gpu or self.device == 0) and step % eval_interval != 0:
                            wandb.log(report_dict)
                        else:
                            report_dict["test/loss"] = avg_test_losses
                            for k,v in avg_test_loss_dicts.items():
                                report_dict[f"test/{k}"] = v
                            wandb.log(report_dict)
                            
                    if (multi_gpu and self.device == 0) and step % save_interval == 0:
                        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                        torch.save(model.module.state_dict(), f"{run_name}/ckpts/loss_{losses}_{timestr}.pt")
                    elif not multi_gpu and step % save_interval == 0:
                        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                        torch.save(model.state_dict(), os.path.join(root_dir, f"ckpts/{run_name}/loss_{losses}_{timestr}.pt"))
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

        trainer = Trainer(device, config)

        trainer.train()
        

if __name__ == "__main__":
    main()




        

