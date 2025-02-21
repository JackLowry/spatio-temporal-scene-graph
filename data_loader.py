import torch
import os
import pickle
import json
import random
from torch.utils.data import Dataset
# import visual_genome.local as vg
import numpy as np
import torchvision
import h5py
from PIL import Image

class SceneGraphDataset(Dataset):
 
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_scenes = len(os.listdir(root_dir))
        self.num_objects = len(os.listdir(os.path.join(root_dir, '0')))
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.num_scenes*self.num_objects

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        object_idx = idx % self.num_objects
        scene_idx = idx // self.num_objects

        item_path = os.path.join(self.root_dir, str(scene_idx), str(object_idx), f"t_{object_idx}.pkl")
        with open(item_path, 'rb') as f:
            sample = pickle.load(f)

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
    
class VisualGenomeDataset(Dataset):
 
    def __init__(self, root_dir, num_objects, scale_factor=1, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.images = h5py.File(os.path.join(root_dir, "imdb_1024.h5"))
        self.scene_graphs = h5py.File(os.path.join(root_dir, "VG-SGG.h5"))
        with open(os.path.join(root_dir, "VG-SGG-dicts.json")) as f:
            self.scene_graph_metadata = json.load(f)

        
        
        self.no_object_label = len(self.scene_graph_metadata["idx_to_label"])
        self.no_relationship_label = len(self.scene_graph_metadata["idx_to_predicate"])
        self.num_object_labels = self.no_object_label+1
        self.num_relationship_labels = self.no_relationship_label+1

        self.num_graphs = self.images["images"].shape[0]
        self.num_objects = num_objects
        self.scale_factor = scale_factor

        #calculate class weights
        label_counts = self.scene_graph_metadata["object_count"]
        idx_to_label = self.scene_graph_metadata["idx_to_label"]
        total_num_labels_including_empty = self.num_graphs*self.num_objects
        label_weights = torch.zeros((self.num_object_labels))

        for i in range(self.num_object_labels-1):
            label_weights[i] = label_counts[idx_to_label[str(i+1)]]
        total_label_count = sum(label_weights)
        label_weights[-1] = total_num_labels_including_empty - total_label_count
        self.label_weights = torch.sum(label_weights)/(self.num_object_labels*label_weights)
        self.label_weights = self.label_weights.cuda()
        # import pdb; pdb.set_trace()
        self.scene_graph_metadata["idx_to_label"]["151"] = "None"

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):


        image = self.images["images"][idx]
        image = torch.Tensor(image)
        desired_size = (round(image.shape[-2]*self.scale_factor), round(image.shape[-1]*self.scale_factor))
        image = torchvision.transforms.functional.resize(image, desired_size)

        image_shape = image.shape

        # import pdb; pdb.set_trace()
        
        image = self.transform[0](image)
        
        required_padding = [0,0,14-image.shape[-2]%14,14-image.shape[-1]%14]
        if required_padding[-1] != 14:
            image = torchvision.transforms.functional.pad(image, required_padding, 0)
        image = image.cuda()

        start_box_idx = self.scene_graphs["img_to_first_box"][idx]
        end_box_idx = self.scene_graphs["img_to_last_box"][idx]
        start_rel_idx = self.scene_graphs["img_to_first_rel"][idx]
        end_rel_idx = self.scene_graphs["img_to_last_rel"][idx]

        object_idxs = list(range(start_box_idx, end_box_idx+1))
        random.shuffle(object_idxs)
        chosen_object_num = min(end_box_idx+1 - start_box_idx, self.num_objects)
        chosen_object_idxs = object_idxs[:chosen_object_num]

        
        object_data = []
        id_list = []

        for object_idx in chosen_object_idxs:
            
            bbox = self.scene_graphs["boxes_1024"][object_idx]

            bbox_corners = [0,0,0,0]
            bbox_corners[0] = bbox[0] - bbox[2] // 2
            bbox_corners[1] = bbox[1] - bbox[3] // 2
            bbox_corners[2] = bbox[0] + bbox[2] // 2
            bbox_corners[3] = bbox[1] + bbox[3] // 2

            bbox = torch.tensor(bbox_corners)

            # bbox[2] = bbox[0] + bbox[2]
            # bbox[3] = bbox[3] + bbox[1]

            bbox = bbox*self.scale_factor
            bbox = torch.round(bbox)

            # bbox[0] = bbox[0]/image_shape[1]
            # bbox[2] = bbox[2]/image_shape[2]
            # bbox[1] = bbox[1]/image_shape[1]
            # bbox[3] = bbox[3]/image_shape[2]
            

            

            bbox = bbox.unsqueeze(0).cuda()

            object_label = self.scene_graphs["labels"][object_idx]

            id_list.append(object_idx - start_box_idx)
            

            object_data.append({
                "bbox": bbox,
                "object_label": torch.Tensor([object_label.item()])
            })      

        # import pdb; pdb.set_trace()

        while len(object_data) < self.num_objects:
            object_data.append({
                "bbox": torch.zeros((1,4)).cuda(),
                "object_label": torch.Tensor([self.no_object_label])
            })      

        relation_data = [None]*(self.num_objects*(self.num_objects))

        relation_idxs = list(range(start_rel_idx, end_rel_idx+1))

        for relation_idx in relation_idxs:
            # import pdb; pdb.set_trace()

            relationship_label = self.scene_graphs["predicates"][relation_idx]
            relation = self.scene_graphs["relationships"][relation_idx]
            subject_id = relation[0] - start_box_idx
            object_id = relation[1] - start_box_idx
            if subject_id in id_list and object_id in id_list:
                
                subject_idx = id_list.index(subject_id)
                object_idx = id_list.index(object_id)
                subject = object_data[subject_idx]
                object = object_data[object_idx]

                relation_data_idx = subject_idx*self.num_objects + object_idx

                bbox = [min(subject["bbox"][0][0].item(), object["bbox"][0][0].item()),
                        min(subject["bbox"][0][1].item(), object["bbox"][0][1].item()),
                        max(subject["bbox"][0][2].item(), object["bbox"][0][2].item()),
                        max(subject["bbox"][0][3].item(), object["bbox"][0][3].item())]
                bbox = torch.Tensor(bbox).unsqueeze(0).cuda()
                
                subject_center = torch.Tensor([(subject["bbox"][0][2] - subject["bbox"][0][0])//2, (subject["bbox"][0][3] - subject["bbox"][0][1])//2])
                object_center = torch.Tensor([(object["bbox"][0][2] - object["bbox"][0][0])//2, (object["bbox"][0][3] - object["bbox"][0][1])//2])

                dist = subject_center - object_center

                dist[0] = dist[0]/image_shape[1]
                dist[1] = dist[1]/image_shape[2]
                dist = dist.unsqueeze(0).cuda()

                relation_data[relation_data_idx] = {
                    "relationship_label": torch.Tensor([relationship_label.item()]),
                    "bbox": bbox,
                    "dist": dist
                }

        for idx in range(len(relation_data)):
            if relation_data[idx] is None:
                relation_data[idx] = {
                    "relationship_label": torch.Tensor([self.no_relationship_label]),
                    "bbox": torch.zeros((1,4)).cuda(),
                    "dist": torch.zeros([2]).unsqueeze(0).cuda()
                }

        relation_data = [relation_data[i] for i in range(len(relation_data)) if i%self.num_objects != 0]

        object_ret_data  = {
            "bbox": torch.concat([o["bbox"] for o in object_data]),
            "object_label": torch.concat([o["object_label"] for o in object_data])
        }

        relation_ret_data = {
            "relationship_label": torch.concat([e["relationship_label"] for e in relation_data]),
            "bbox": torch.concat([e["bbox"] for e in relation_data]),
            "dist": torch.concat([e["dist"] for e in relation_data])
        }
        # import pdb; pdb.set_trace()

        return_data = {
            "nodes": object_ret_data,
            "edges": relation_ret_data,
            "image": image
        }
        return return_data
                

class StowDataset(Dataset):
 
    def __init__(self, root_dir, num_objects, scale_factor=1, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.images = h5py.File(os.path.join(root_dir, "sg_data.h5"))
        with open(os.path.join(root_dir, "sg_data.pkl"), 'rb') as f:
            self.scene_graphs = pickle.load(f)
        with open(os.path.join(root_dir, "sg_data.json"), 'r') as f:
            self.scene_graph_metadata = json.load(f)

        
        
        self.no_object_label = len(list(self.scene_graph_metadata["object_metadata"].keys()))
        self.no_relationship_label = 0
        self.num_object_labels = self.no_object_label+1
        self.num_relationship_labels = len((self.scene_graph_metadata["edge_metadata"].keys()))
        self.scene_graph_metadata["object_id_to_name"][str(self.no_object_label)] = "None"

        self.num_graphs = self.images["rgba"].shape[0]*2
        self.num_objects = num_objects
        self.scale_factor = scale_factor

        #calculate class weights
        # label_counts = self.scene_graph_metadata["object_count"]
        # idx_to_label = self.scene_graph_metadata["idx_to_label"]
        # total_num_labels_including_empty = self.num_graphs*self.num_objects
        # label_weights = torch.zeros((self.num_object_labels))

        # for i in range(self.num_object_labels-1):
        #     label_weights[i] = label_counts[idx_to_label[str(i+1)]]
        # total_label_count = sum(label_weights)
        # label_weights[-1] = total_num_labels_including_empty - total_label_count
        # self.label_weights = torch.sum(label_weights)/(self.num_object_labels*label_weights)
        # self.label_weights = self.label_weights.cuda()
        # import pdb; pdb.set_trace()
        # self.scene_graph_metadata["idx_to_label"]["151"] = "None"

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):

        graph_idx = idx // 2
        pair_idx = idx % 2


        image = self.images["rgba"][graph_idx][pair_idx][:, :, :3]
        
        image = torch.Tensor(image)

        graph = self.scene_graphs[graph_idx][pair_idx]

        #channel first
        image = image.permute((2,0,1))
        orig_image = image.clone().to(torch.uint8)
        # resize
        if self.scale_factor != 1:
            desired_size = (round(image.shape[-2]*self.scale_factor), round(image.shape[-1]*self.scale_factor))
            image = torchvision.transforms.functional.resize(image, desired_size)
        
        #apply transformations, don't understand why [0] is required
        image = self.transform[0](image)
    
        required_padding = [0,0,14-image.shape[-2]%14,14-image.shape[-1]%14]
        if required_padding[-1] != 14:
            image = torchvision.transforms.functional.pad(image, required_padding, 0)

        image = image.cuda()

        object_to_idx = {}

        object_to_training_idxs = {}
        training_idx_counter = 0

        object_data = []
        for object in graph["nodes"].keys():
            
            bbox = graph["nodes"][object]["bbox"]
            bbox = torch.tensor(bbox)

            #scale according to scale factor
            bbox = bbox*self.scale_factor
            bbox = torch.round(bbox)
            bbox = bbox.unsqueeze(0).cuda()

            object_label = graph["nodes"][object]["object_id"]            

            object_data.append({
                "bbox": bbox.to(torch.float),
                "object_label": torch.Tensor([object_label]).to(torch.int32)
            })      

            object_to_idx[object] = len(object_data) - 1

            object_to_training_idxs[object] = training_idx_counter
            training_idx_counter += 1 

        while len(object_data) < self.num_objects:
            object_data.append({
                "bbox": torch.zeros((1,4)).cuda().to(torch.float),
                "object_label": torch.Tensor([self.no_object_label])
            })      

        relation_data = [None]*(self.num_objects*(self.num_objects))

        for relation_tuple in graph["edges"].keys():
            # import pdb; pdb.set_trace()

            relationship_label = graph["edges"][relation_tuple]["name"]
            relation = graph["edges"][relation_tuple]["relation_id"]
            subject_id = relation_tuple[0]
            object_id = relation_tuple[1]
                
            subject = relation_tuple[0]
            object = relation_tuple[1]

            relation_data_idx = object_to_training_idxs[subject]*self.num_objects + object_to_idx[object]

            bbox = graph["edges"][relation_tuple]["bbox"]
            bbox = torch.Tensor(bbox).unsqueeze(0).cuda()
            bbox = bbox*self.scale_factor
            bbox = torch.round(bbox)

            dist = graph["edges"][relation_tuple]["xyz_offset"]
            dist = torch.Tensor(dist).squeeze().unsqueeze(0).cuda()

            relation_data[relation_data_idx] = {
                "relationship_label": torch.Tensor([relation]),
                "bbox": bbox.to(torch.float),
                "dist": dist
            }

        for idx in range(len(relation_data)):
            if relation_data[idx] is None:
                relation_data[idx] = {
                    "relationship_label": torch.Tensor([self.no_relationship_label]),
                    "bbox": torch.zeros((1,4)).cuda().to(torch.float),
                    "dist": torch.zeros([3]).unsqueeze(0).cuda()
                }

        relation_data = [relation_data[i] for i in range(len(relation_data)) if i%self.num_objects != 0]

        object_ret_data  = {
            "bbox": torch.concat([o["bbox"] for o in object_data]),
            "object_label": torch.concat([o["object_label"] for o in object_data])
        }

        relation_ret_data = {
            "relationship_label": torch.concat([e["relationship_label"] for e in relation_data]),
            "bbox": torch.concat([e["bbox"] for e in relation_data]),
            "dist": torch.concat([e["dist"] for e in relation_data]),
        }

        return_data = {
            "nodes": object_ret_data,
            "edges": relation_ret_data,
            "image": image,
            "orig_image": orig_image
        }
        return return_data
                

class IsaacLabDataset(Dataset):
 
    def __init__(self, root_dir, scale_factor=1, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.num_scenes = len(os.listdir(root_dir)) - 1
        self.num_objects = len(os.listdir(os.path.join(root_dir, '0')))

        # self.images = h5py.File(os.path.join(root_dir, "sg_data.h5"))
        # with open(os.path.join(root_dir, "sg_data.pkl"), 'rb') as f:
        #     self.scene_graphs = pickle.load(f)
        # with open(os.path.join(root_dir, "sg_data.json"), 'r') as f:
        #     self.scene_graph_metadata = json.load(f)

        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.metadata["object_id_to_name"] = {
            v['id']:k for k,v in self.metadata["node_data"].items()
        }
        
        self.no_object_label = 0
        self.no_relationship_label = 0
        self.num_object_labels = len(list(self.metadata["node_data"].keys()))
        self.num_relationship_labels = 7#len(list(self.metadata["edge_data"].keys()))

        self.num_graphs = self.num_scenes * self.num_objects
        self.scale_factor = scale_factor

        #calculate class weights
        # label_counts = self.scene_graph_metadata["count"]
        # total_num_labels_including_empty = self.num_graphs*self.num_objects
        # label_weights = torch.zeros((self.num_object_labels))

        # for i in range(self.num_object_labels-1):
        #     label_weights[i] = label_counts[idx_to_label[str(i+1)]]
        # total_label_count = sum(label_weights)
        # label_weights[-1] = total_num_labels_including_empty - total_label_count
        # self.label_weights = torch.sum(label_weights)/(self.num_object_labels*label_weights)
        # self.label_weights = self.label_weights.cuda()
        # import pdb; pdb.set_trace()
        # self.scene_graph_metadata["idx_to_label"]["151"] = "None"

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):

        object_idx = idx % self.num_objects
        scene_idx = idx // self.num_objects

        item_path = os.path.join(self.root_dir, str(scene_idx), f"t_{object_idx}.pkl")
        with open(item_path, 'rb') as f:
            sample = pickle.load(f)



        image = sample["images"]["rgb"][0].to(torch.float32)/255
        
        image = torch.Tensor(image)

        graph = sample["graph"][0]

        #channel first
        image = image.permute((2,0,1))
        orig_image = image.clone()
        image = image.to(torch.float32)
        # resize
        if self.scale_factor != 1:
            desired_size = (round(image.shape[-2]*self.scale_factor), round(image.shape[-1]*self.scale_factor))
            image = torchvision.transforms.functional.resize(image, desired_size)
        
        #apply transformations, don't understand why [0] is required
        image = self.transform[0](image)
    
        image = image.cuda()

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
            bbox = bbox.unsqueeze(0).cuda()

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
                "bbox": torch.zeros((1,4)).cuda().to(torch.float),
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

            relation_data_idx = object_to_training_idxs[subject]*self.num_objects + object_to_idx[object]
            edge_idx_to_node_idxs[relation_data_idx] = [relation_data_idx, object_to_idx[subject_id], object_to_idx[object_id]]
            if node_network_mask[object_to_idx[subject_id]] == 0 or node_network_mask[object_to_idx[object_id]] == 0:
                edge_network_mask[relation_data_idx] = False
            else:
                edge_network_mask[relation_data_idx] = True

            bbox = graph["edges"][relation_tuple]["bbox"]
            bbox = torch.Tensor(bbox).unsqueeze(0).cuda()
            bbox = bbox*self.scale_factor
            bbox = torch.round(bbox)

            dist = graph["edges"][relation_tuple]["xyz_offset"]
            dist = torch.Tensor(dist).squeeze().unsqueeze(0).cuda()

            relation_data[relation_data_idx] = {
                "relationship_label": torch.Tensor([relation]),
                "bbox": bbox.to(torch.float),
                "dist": dist
            }

        edge_network_mask = [edge_network_mask[i] for i in range(len(edge_network_mask)) if relation_data[i]  is not None]
        relation_data = [relation_data[i] for i in range(len(relation_data)) if relation_data[i] is not None]
        edge_idx_to_node_idxs = [edge_idx_to_node_idxs[i] for i in range(len(edge_idx_to_node_idxs)) if edge_idx_to_node_idxs[i] is not None]
        edge_idx_to_node_idxs = torch.Tensor(edge_idx_to_node_idxs)
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

        return_data = {
            "nodes": object_ret_data,
            "edges": relation_ret_data,
            "image": image,
            "orig_image": orig_image,
            "edge_idx_to_node_idxs": edge_idx_to_node_idxs,
            "node_network_mask": node_network_mask,
            "edge_network_mask": edge_network_mask
        }
        return return_data
    
class IsaacLabTemporalDataset(Dataset):
 
    def __init__(self, root_dir, scale_factor=1, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.num_scenes = len(os.listdir(root_dir)) - 1
        self.num_objects = len(os.listdir(os.path.join(root_dir, '0')))

        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        self.metadata["object_id_to_name"] = {
            v['id']:k for k,v in self.metadata["node_data"].items()
        }
        
        self.no_object_label = 0
        self.no_relationship_label = 0
        self.num_object_labels = len(list(self.metadata["node_data"].keys()))
        self.num_relationship_labels = 7#len(list(self.metadata["edge_data"].keys()))

        self.num_graphs = self.num_scenes * self.num_objects
        self.scale_factor = scale_factor

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, scene_idx):

        sequence_object_ret_data = []
        sequence_relation_ret_data = []
        sequence_image = []
        sequence_orig_image = []
        sequence_edge_idx_to_node_idxs = []
        sequence_node_network_mask = []
        sequence_edge_network_mask = []

        for object_idx in range(self.num_objects):
            item_path = os.path.join(self.root_dir, str(scene_idx), f"t_{object_idx}.pkl")
            with open(item_path, 'rb') as f:
                sample = pickle.load(f)



            image = sample["images"]["rgb"][0].to(torch.float32)/255
            
            image = torch.Tensor(image)

            graph = sample["graph"][0]

            #channel first
            image = image.permute((2,0,1))
            orig_image = image.clone()
            image = image.to(torch.float32)
            # resize
            if self.scale_factor != 1:
                desired_size = (round(image.shape[-2]*self.scale_factor), round(image.shape[-1]*self.scale_factor))
                image = torchvision.transforms.functional.resize(image, desired_size)
            
            #apply transformations, don't understand why [0] is required
            image = self.transform[0](image)
        
            image = image.cuda()

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
                bbox = bbox.unsqueeze(0).cuda()

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
                    "bbox": torch.zeros((1,4)).cuda().to(torch.float),
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

                relation_data_idx = object_to_training_idxs[subject]*self.num_objects + object_to_idx[object]
                edge_idx_to_node_idxs[relation_data_idx] = [relation_data_idx, object_to_idx[subject_id], object_to_idx[object_id]]
                if node_network_mask[object_to_idx[subject_id]] == 0 or node_network_mask[object_to_idx[object_id]] == 0:
                    edge_network_mask[relation_data_idx] = False
                else:
                    edge_network_mask[relation_data_idx] = True

                bbox = graph["edges"][relation_tuple]["bbox"]
                bbox = torch.Tensor(bbox).unsqueeze(0).cuda()
                bbox = bbox*self.scale_factor
                bbox = torch.round(bbox)

                dist = graph["edges"][relation_tuple]["xyz_offset"]
                dist = torch.Tensor(dist).squeeze().unsqueeze(0).cuda()

                relation_data[relation_data_idx] = {
                    "relationship_label": torch.Tensor([relation]),
                    "bbox": bbox.to(torch.float),
                    "dist": dist
                }

            edge_network_mask = [edge_network_mask[i] for i in range(len(edge_network_mask)) if relation_data[i]  is not None]
            relation_data = [relation_data[i] for i in range(len(relation_data)) if relation_data[i] is not None]
            edge_idx_to_node_idxs = [edge_idx_to_node_idxs[i] for i in range(len(edge_idx_to_node_idxs)) if edge_idx_to_node_idxs[i] is not None]
            edge_idx_to_node_idxs = torch.Tensor(edge_idx_to_node_idxs)
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
            sequence_object_ret_data.append(object_ret_data)
            sequence_relation_ret_data.append(relation_ret_data)
            sequence_image.append(image)
            sequence_orig_image.append(orig_image)
            sequence_edge_idx_to_node_idxs.append(edge_idx_to_node_idxs)
            sequence_node_network_mask.append(node_network_mask)
            sequence_edge_network_mask.append(edge_network_mask)

        #stack bboxes and labels together
        sequence_object_ret_data_stacked = {}
        for key in sequence_object_ret_data[0].keys():
            sequence_object_ret_data_stacked[key] = torch.stack(
                [sequence_object_ret_data[i][key] for i in range(len(sequence_object_ret_data))]
            )

        sequence_relation_ret_data_stacked = {}
        for key in sequence_relation_ret_data[0].keys():
            sequence_relation_ret_data_stacked[key] = torch.stack(
                [sequence_relation_ret_data[i][key] for i in range(len(sequence_relation_ret_data))]
            )


        return_data = {
            "nodes": sequence_object_ret_data_stacked,
            "edges": sequence_relation_ret_data_stacked,
            "image": torch.stack(sequence_image),
            "orig_image": torch.stack(sequence_orig_image),
            "edge_idx_to_node_idxs": torch.stack(sequence_edge_idx_to_node_idxs),
            "node_network_mask": torch.stack(sequence_node_network_mask),
            "edge_network_mask": torch.stack(sequence_edge_network_mask)
        }
        return return_data