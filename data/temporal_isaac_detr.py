import torch
import os
import pickle
import json
import random
from torch.utils.data import Dataset
# import visual_genome.local as vg
import numpy as np
import torchvision
from PIL import Image
import tqdm

from util.box_ops import box_xyxy_to_cxcywh

class TemporalIsaacLabDetrDataset(Dataset):
 
    def __init__(self, root_dir,
                 feature_extractor, 
                 scale_factor=1.0, 
                 transform=None,
                 num_object_queries=50):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.feature_extractor = feature_extractor
        
        self.root_dir = root_dir
        self.transform = transform

        self.num_scenes = len(os.listdir(root_dir)) - 2
        self.num_objects = len(os.listdir(os.path.join(root_dir, '0')))

        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.metadata["object_id_to_name"] = {
            v['id']:k for k,v in self.metadata["node_data"].items()
        }

        # self.metadata['edge_name_to_id'] = {
        #     "no_relation": 0,
        #     "in_front_of": 1,
        #     "behind": 2,
        #     "on_top_of": 3,
        #     "below": 4,
        #     "right_of": 5,
        #     "left_of": 6,
        # }

        self.metadata['edge_name_to_id'] = {
            "in_front_of": 0,
            "behind": 1,
            "on_top_of": 2,
            "below": 3,
            "right_of": 4,
            "left_of": 5,
        }
        self.metadata["edge_id_to_name"] = {
            v:k for k,v in self.metadata["edge_name_to_id"].items()
        }

        self.metadata["edge_id_to_name"]
        
        self.no_object_label = 0
        self.no_relationship_label = 0
        self.num_object_labels = len(list(self.metadata["node_data"].keys()))
        self.num_relationship_labels = 6#len(list(self.metadata["edge_data"].keys()))

        self.scale_factor = scale_factor
        self.num_object_queries = num_object_queries

    def __len__(self):
        return self.num_scenes
    
    def load_idx(self, scene_idx, object_idx):
        item_path = os.path.join(self.root_dir, str(scene_idx), f"t_{object_idx}_cpu.pkl")
        with open(item_path, 'rb') as f:
            sample = torch.load(f)
        return sample

    def __getitem__(self, scene_idx):

        sequence_pixel_values = []
        sequence_target = []
        sequence_orig_img = []

        for object_idx in range(self.num_objects):
            sample = self.load_idx(scene_idx, object_idx)

            image = sample["images"]["rgb"][0].to(torch.float32).cpu()/255
            
            image = torch.Tensor(image)

            graph = sample["graph"][0]

            object_to_idx = {}

            object_to_training_idxs = {}
            training_idx_counter = 0

            object_data = []
            node_network_mask = []
            for object in graph["nodes"].keys():
                
                bbox = graph["nodes"][object]["bbox"]
                bbox = torch.tensor(bbox)

                #scale according to scale factor
                bbox = bbox*self.scale_factor
                bbox = torch.round(bbox)
                bbox = bbox.unsqueeze(0)

                object_label = self.metadata['node_data'][graph["nodes"][object]["class_name"]]['id']     


                object_data.append({
                    "bbox": bbox.to(torch.float),
                    "object_label": torch.Tensor([object_label]).to(torch.int32)
                })     
                if object_label == self.metadata['node_data']["None"]['id']:
                    node_network_mask.append(0)
                else:
                    node_network_mask.append(1) 

                object_to_idx[object] = len(object_data) - 1

                object_to_training_idxs[object] = training_idx_counter
                training_idx_counter += 1 

            annotations = []

            idx = scene_idx*self.num_objects + object_idx

            for o in object_data:
                bbox = box_xyxy_to_cxcywh(o["bbox"]).squeeze().numpy()
                area = bbox[-1]*bbox[-2] # width * height
                annotation = {
                    "image_id": idx,
                    "bbox": bbox,
                    "area": area,
                    "category_id": 0
                } 
                annotations.append(annotation)


            relationships = []
            for relation_tuple in graph["edges"].keys():
                (subject_name, object_name) = relation_tuple
                relation_id = graph["edges"][relation_tuple]['relation_id']
                if relation_id == 0: #ignore no_relation, we don't want to predict these
                    continue
                relation_id = relation_id - 1
                subject_id = object_to_training_idxs[subject_name]
                object_id  = object_to_training_idxs[object_name]
                relationships.append(torch.Tensor([subject_id, object_id, relation_id]))

            target = {
                "image_id": idx,
                "annotations": annotations,
            }
            encoding = self.feature_extractor(
                image, target, return_tensors="pt"
            )
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
            target = encoding["labels"][0]  # remove batch dimension
            if len(relationships) == 0:
                target["rel"] = []
            else:
                target["rel"] = torch.stack(relationships)

            sequence_pixel_values.append(pixel_values)
            sequence_target.append(target)
            sequence_orig_img.append(image)

        
        # return_data = {
        #     "nodes": object_ret_data,
        #     "edges": relation_ret_data,
        #     "image": torch.stack(sequence_image),
        #     "orig_image": torch.stack(sequence_orig_image),
        #     "edge_idx_to_node_idxs": torch.stack(sequence_edge_idx_to_node_idxs),
        #     "node_network_mask": torch.stack(sequence_node_network_mask),
        #     "edge_network_mask": torch.stack(sequence_edge_network_mask)
        # }
        return (sequence_pixel_values, sequence_orig_img), sequence_target

    @staticmethod
    def get_statistics(data_subset: torch.utils.data.dataset.Subset, cache_filename="train_stats.npy", force_calculate=False):
        dataset = data_subset.dataset
        cache_path = os.path.join(dataset.root_dir, cache_filename)
        if not force_calculate and os.path.isfile(cache_path):
            return np.load(cache_path)
        
        fg_matrix = np.zeros(
            (
                dataset.num_object_labels,
                dataset.num_object_labels,
                dataset.num_relationship_labels,
            ),
            dtype=np.int64,
        )

        # rel = train_data.rel
        for scene_idx in tqdm.tqdm(data_subset.indices):
            for object_idx in range(dataset.num_objects):
                sample = dataset.load_idx(scene_idx, object_idx)

                graph = sample["graph"][0]

                for relation_tuple in graph["edges"].keys():
                    (subject_name, object_name) = relation_tuple
                    relation_id = graph["edges"][relation_tuple]['relation_id']
                    if relation_id == 0: #ignore no_relation, we don't want to predict these
                        continue
                    relation_id = relation_id - 1
                    subject_id = dataset.metadata["node_data"][graph["nodes"][subject_name]["class_name"]]["id"]
                    object_id  = dataset.metadata["node_data"][graph["nodes"][object_name]["class_name"]]["id"]
                    fg_matrix[subject_id, object_id, relation_id] += 1

        np.save(cache_path, fg_matrix)

        return fg_matrix
    
    
