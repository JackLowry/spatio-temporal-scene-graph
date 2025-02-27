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

config_name = "networks/temporal/two.yaml"

def multi_gpu_train(rank, world_size, config):
    ddp_setup(rank, world_size)
    train(rank, config)

def train(device, config):
    wandb.init(project="scenegraphgeneration-isaac", group="DDP", config=config)
    run_name = f'{wandb.run.name}_{wandb.run.id}'
    generator1 = torch.Generator().manual_seed(42)
    multi_gpu = config["multi_gpu"]

    #process dataset
    # root_data_dir = "/mmfs1/home/jrl712/amazon_home/data/isaaclab_sg_1000"
    # root_dir = "/mmfs1/home/jrl712/amazon_home/scene_graph/spatio-temporal-scene-graph"
    # root_data_dir = "/home/jack/research/data/isaaclab_sg/01-31-2025:14-46-04"
    # root_dir = "/home/jack/research/scene_graph/spatio_temporal_sg"
    root_data_dir = config['root_data_dir']
    root_dir = config['root_dir']
    os.mkdir(os.path.join(root_dir, 'ckpts', run_name))
    preproccess = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    scale_factor = config['scale_factor']
    batch_size = config['batch_size']
    dataset = IsaacLabTemporalDataset(root_data_dir, scale_factor=scale_factor, transform=preproccess)
    

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.9, .1], generator=generator1,)

    if multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    training_iterations = 1000

    network_config = config["network_config"]
    model = TemporalSceneGraphModel(batch_size, dataset.num_objects, dataset.num_object_labels, dataset.num_relationship_labels, dataset.num_objects, network_config)
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

    # edge_idx_to_node_idxs = []
    # for i in range(dataset.num_objects):
    #     for j in range(dataset.num_objects):
    #         if i == j:
    #             continue
    #         edge_idx_to_node_idxs.append(
    #             [i*dataset.num_objects + j, i, j]
    #         )

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
            batch_edge_idx_to_node_idxs = batch["edge_idx_to_node_idxs"].to(device)
            gt_node_label = batch["nodes"]["object_label"].to(device)
            gt_edge_label = batch["edges"]["relationship_label"].to(device)
            gt_pos = batch["edges"]["dist"].to(device)
            
            batch_node_network_mask = batch['node_network_mask'].to(device)
            batch_edge_network_mask = batch['edge_network_mask'].to(device)

            # import pdb; pdb.set_trace
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

            node_labels, edge_labels, relative_position = model(batch_images, 
                                                                batch_object_bbox, 
                                                                batch_union_bbox, 
                                                                batch_edge_idx_to_node_idxs,
                                                                batch_node_network_mask,
                                                                batch_edge_network_mask)

            node_labels = node_labels.reshape(-1, node_labels.shape[-1])
            gt_node_label = gt_node_label.reshape(-1)
            batch_node_network_mask = batch_node_network_mask.reshape(-1)
            
            edge_labels = edge_labels.reshape(-1, edge_labels.shape[-1])
            gt_edge_label = gt_edge_label.reshape(-1)
            batch_edge_network_mask = batch_edge_network_mask.reshape(-1)


            relative_position = relative_position.reshape(-1, relative_position.shape[-1])
            gt_pos = gt_pos.reshape(-1, gt_pos.shape[-1])



            node_loss = node_label_loss_metric(node_labels[batch_node_network_mask], gt_node_label.long()[batch_node_network_mask])
            edge_loss = edge_label_loss_metric(edge_labels[batch_edge_network_mask], gt_edge_label.long()[batch_edge_network_mask])
            edge_loss[torch.isnan(edge_loss)] = 0.0
            pos_loss = pos_loss_metric(relative_position[batch_edge_network_mask], gt_pos[batch_edge_network_mask])
            pos_loss[torch.isnan(pos_loss)] = 0.0

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
                        node_logits_total = []
                        edge_logits_total = []
                        node_gt_total = []
                        edge_gt_total = []
                        num_edge_samples = 0
                        
                        for test_batch in (pbar := tqdm(test_dataloader, leave=False)):

                            batch_rgb = batch["image"].to(device).to(torch.float)
                            batch_images = batch_rgb
                            batch_object_bbox = batch["nodes"]["bbox"].to(device)
                            batch_union_bbox = batch["edges"]["bbox"].to(device)
                            batch_edge_idx_to_node_idxs = batch["edge_idx_to_node_idxs"].to(device)
                            gt_node_label = batch["nodes"]["object_label"].to(device)
                            gt_edge_label = batch["edges"]["relationship_label"].to(device)
                            gt_pos = batch["edges"]["dist"].to(device)
            
                            batch_node_network_mask = batch['node_network_mask'].to(device)
                            batch_edge_network_mask = batch['edge_network_mask'].to(device)

                            if config["randomize_input"] and config["randomize_ground_truth"]:
                                (batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs, node_parameters, edge_parameters) \
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

                            node_labels, edge_labels, relative_position = model(batch_images, batch_object_bbox, batch_union_bbox, batch_edge_idx_to_node_idxs,
                                                                                batch_node_network_mask, batch_edge_network_mask)
                            node_labels = node_labels.reshape(-1, node_labels.shape[-1])
                            gt_node_label = gt_node_label.reshape(-1)
                            batch_node_network_mask = batch_node_network_mask.reshape(-1)
                            
                            edge_labels = edge_labels.reshape(-1, edge_labels.shape[-1])
                            gt_edge_label = gt_edge_label.reshape(-1)
                            batch_edge_network_mask = batch_edge_network_mask.reshape(-1)


                            relative_position = relative_position.reshape(-1, relative_position.shape[-1])
                            gt_pos = gt_pos.reshape(-1, gt_pos.shape[-1])

                            node_loss = node_label_loss_metric(node_labels[batch_node_network_mask], gt_node_label.long()[batch_node_network_mask])
                            edge_loss = edge_label_loss_metric(edge_labels[batch_edge_network_mask], gt_edge_label.long()[batch_edge_network_mask])
                            edge_loss[torch.isnan(edge_loss)] = 0.0
                            pos_loss = pos_loss_metric(relative_position[batch_edge_network_mask], gt_pos[batch_edge_network_mask])
                            pos_loss[torch.isnan(pos_loss)] = 0.0


                            pred_node_logits = nn.functional.softmax(node_labels[batch_node_network_mask], dim=-1)
                            node_logits_total.append(pred_node_logits.cpu())

                            node_gt_total.append(gt_node_label[batch_node_network_mask].cpu().to(torch.long))

                            #only compute f1 score if there are valid edges
                            if batch_edge_network_mask.any():
                                pred_edge_logits = nn.functional.softmax(edge_labels[batch_edge_network_mask], dim=-1)
                                edge_logits_total.append(pred_edge_logits.cpu())
                                edge_gt_total.append(gt_edge_label[batch_edge_network_mask].cpu().to(torch.long))

                            test_node_loss.append(node_loss)
                            test_edge_loss.append(edge_loss)
                            test_dist_loss.append(pos_loss)
                            num_edge_samples += batch_edge_network_mask.sum().cpu().item()
                            test_loss.append(pos_loss)
                        avg_test_loss = sum(test_loss)/len(test_loss)
                        avg_test_node_loss = sum(test_node_loss)/len(test_node_loss)
                        avg_test_edge_loss = sum(test_edge_loss)/len(test_edge_loss)
                        avg_test_pos_loss = sum(test_dist_loss)/num_edge_samples

                        node_logits_total = torch.concat(node_logits_total)
                        edge_logits_total = torch.concat(edge_logits_total)
                        node_gt_total = torch.concat(node_gt_total)
                        edge_gt_total = torch.concat(edge_gt_total)

                        node_f1 = metrics.graph_f1_score(node_logits_total, node_gt_total)
                        edge_f1 = metrics.graph_f1_score(edge_logits_total, edge_gt_total)
                        node_recall_at_5 = metrics.graph_recall_at_k(node_logits_total, node_gt_total, 5)
                        # node_language_labels = [dataset.metadata["object_id_to_name"][idx] for idx in range(dataset.num_object_labels)]
                        # node_confusion_matrix = metrics.confusion_matrix(node_logits_total, node_gt_total, labels=node_language_labels)
                        # edge_confusion_matrix = metrics.confusion_matrix(edge_logits_total, edge_gt_total, labels=edge_language_labels)


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

                node_labels_viz = node_labels.reshape(node_labels.shape[0]//(dataset.num_objects**2), dataset.num_object_labels, dataset.num_objects, -1)
                node_labels_viz = node_labels_viz[0, -1]
                node_labels_viz = torch.argmax(nn.functional.softmax(node_labels_viz, dim=-1), dim=-1)
                img_language_labels = [dataset.metadata["object_id_to_name"][idx.cpu().item()] for idx in node_labels_viz]
                img = draw_image(batch["orig_image"][0][-1], batch_object_bbox.cpu()[0][-1]*(1/scale_factor), img_language_labels)
                img.save("test.png")
                img = wandb.Image(img)
                if (not multi_gpu or device == 0) and step % eval_interval != 0:
                    wandb.log({"train/loss": avg_loss, "train/node_loss": avg_node_loss, "train/edge_loss": avg_edge_loss, "train/pos_loss": pos_loss, "img": img})
                else:
                    wandb.log({"test/node_f1_score": node_f1, "test/edge_f1_score": edge_f1,
                               "test/node_recall@5": node_recall_at_5,
                            "train/loss": avg_loss, "train/node_loss": avg_node_loss, "train/edge_loss": avg_edge_loss, "train/pos_loss": pos_loss, 
                            "test/loss": avg_test_loss, "test/node_loss": avg_test_node_loss, "test/edge_loss": avg_test_edge_loss, 
                            "test/pos_loss": avg_test_pos_loss, "img": img})
                    
            if (multi_gpu and device == 0) and step % save_interval == 0:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                torch.save(model.module.state_dict(), f"{run_name}/ckpts/loss_{avg_loss}_{timestr}.pt")
            elif not multi_gpu and step % save_interval == 0:
                timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
                torch.save(model.state_dict(), os.path.join(root_dir, f"ckpts/{run_name}/loss_{avg_loss}_{timestr}.pt"))
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
        train(device, config)
        

if __name__ == "__main__":
    main()




        

