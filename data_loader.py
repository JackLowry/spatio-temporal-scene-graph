import torch
import os
import pickle
from torch.utils.data import Dataset

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
        
