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

from util.box_ops import box_xyxy_to_cxcywh

class IsaacLabDetrDataset(Dataset):
 
    def __init__(self, root_dir,
                 feature_extractor, 
                 scale_factor=1, 
                 transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.feature_extractor = feature_extractor
        
        self.root_dir = root_dir
        self.transform = transform

        self.num_scenes = len(os.listdir(root_dir)) - 1
        self.num_objects = len(os.listdir(os.path.join(root_dir, '0')))

        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.metadata["object_id_to_name"] = {
            v['id']:k for k,v in self.metadata["node_data"].items()
        }

        self.metadata['edge_name_to_id'] = {
            "no_relation": 0,
            "in_front_of": 1,
            "behind": 2,
            "on_top_of": 3,
            "below": 4,
            "right_of": 5,
            "left_of": 6,
        }
        self.metadata["edge_id_to_name"] = {
            v:k for k,v in self.metadata["edge_name_to_id"].items()
        }

        self.metadata["edge_id_to_name"]
        
        self.no_object_label = 0
        self.no_relationship_label = 0
        self.num_object_labels = len(list(self.metadata["node_data"].keys()))
        self.num_relationship_labels = 7#len(list(self.metadata["edge_data"].keys()))

        self.num_graphs = self.num_scenes * self.num_objects
        self.scale_factor = scale_factor

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):

        sequence_object_ret_data = []
        sequence_relation_ret_data = []
        sequence_image = []
        sequence_orig_image = []
        sequence_edge_idx_to_node_idxs = []
        sequence_node_network_mask = []
        sequence_edge_network_mask = []

        scene_idx = idx // self.num_objects
        object_idx = idx % self.num_objects

        item_path = os.path.join(self.root_dir, str(scene_idx), f"t_{object_idx}.pkl")
        with open(item_path, 'rb') as f:
            sample = pickle.load(f)#, map_location='cpu')



        image = sample["images"]["rgb"][0].to(torch.float32).cpu()/255
        
        image = torch.Tensor(image)

        graph = sample["graph"][0]

        #channel first

        # resize

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

        while len(object_data) < self.num_objects:
            object_data.append({
                "bbox": torch.zeros((1,4)).to(torch.float),
                "object_label": torch.Tensor([self.no_object_label])
            })      
            node_network_mask.append(0)

        node_network_mask = torch.Tensor(node_network_mask) == 1

        relation_data = [None]*(self.num_objects*(self.num_objects))

        edge_idx_to_node_idxs = [None]*(self.num_objects*(self.num_objects))

        edge_network_mask = [False]*(self.num_objects*(self.num_objects))

        for relation_tuple in graph["edges"].keys():
            # import pdb; pdb.set_trace()

            relationship_label = graph["edges"][relation_tuple]["name"]
            relation = graph["edges"][relation_tuple]["relation_id"]
            subject_id = relation_tuple[0]
            object_id = relation_tuple[1]
                
            subject = relation_tuple[0]
            object = relation_tuple[1]

            bbox = graph["edges"][relation_tuple]["bbox"]
            bbox = torch.Tensor(bbox).unsqueeze(0)
            bbox = bbox*self.scale_factor
            bbox = torch.round(bbox)

            relation_data_idx = object_to_training_idxs[subject]*self.num_objects + object_to_idx[object]
            edge_idx_to_node_idxs[relation_data_idx] = [relation_data_idx, object_to_idx[subject_id], object_to_idx[object_id]]
            if node_network_mask[object_to_idx[subject_id]] == 0 or node_network_mask[object_to_idx[object_id]] == 0:
                edge_network_mask[relation_data_idx] = False
                bbox[:] = 0
            else:
                edge_network_mask[relation_data_idx] = True


            dist = graph["edges"][relation_tuple]["xyz_offset"]
            dist = torch.Tensor(dist).squeeze().unsqueeze(0)

            relation_data[relation_data_idx] = {
                "relationship_label": torch.Tensor([relation]),
                "bbox": bbox.to(torch.float),
                "dist": dist
            }

        edge_network_mask = [edge_network_mask[i] for i in range(len(edge_network_mask)) if relation_data[i]  is not None]
        relation_data = [relation_data[i] for i in range(len(relation_data)) if relation_data[i] is not None]
        edge_idx_to_node_idxs = [edge_idx_to_node_idxs[i] for i in range(len(edge_idx_to_node_idxs)) if edge_idx_to_node_idxs[i] is not None]
        edge_idx_to_node_idxs = torch.Tensor(edge_idx_to_node_idxs)
        edge_idx_to_node_idxs[:, 0] = edge_idx_to_node_idxs[:, 0] - (1+edge_idx_to_node_idxs[:, 0]//(self.num_objects+1))
        edge_network_mask = torch.Tensor(edge_network_mask) == 1

        object_ret_data  = {
            "bbox": torch.concat([o["bbox"] for o in object_data]),
            "object_label": torch.concat([o["object_label"] for o in object_data])
        }

        relation_ret_data = {
            "relationship_label": torch.concat([e["relationship_label"] for e in relation_data]),
            "bbox": torch.concat([e["bbox"] for e in relation_data]),
            "dist": torch.concat([e["dist"] for e in relation_data]),
        }

        annotations = []

        for o in object_data:
            bbox = box_xyxy_to_cxcywh(o["bbox"]).squeeze().numpy()
            area = bbox[-1]*bbox[-2]
            annotation = {
                "image_id": idx,
                "bbox": bbox,
                "area": area,
                "category_id": 0
            } 
            annotations.append(annotation)

        target = {
            "image_id": idx,
            "annotations": annotations
        }
        encoding = self.feature_extractor(
            image, target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        # return_data = {
        #     "nodes": object_ret_data,
        #     "edges": relation_ret_data,
        #     "image": torch.stack(sequence_image),
        #     "orig_image": torch.stack(sequence_orig_image),
        #     "edge_idx_to_node_idxs": torch.stack(sequence_edge_idx_to_node_idxs),
        #     "node_network_mask": torch.stack(sequence_node_network_mask),
        #     "edge_network_mask": torch.stack(sequence_edge_network_mask)
        # }
        return pixel_values, target 