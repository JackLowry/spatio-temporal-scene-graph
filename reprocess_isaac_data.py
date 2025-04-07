import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

num_processes = 25

def process(process_id):
    root_dir = "/mmfs1/home/jrl712/amazon_home/data/isaaclab_sg_5000_300_obj"
    num_scenes = (len(os.listdir(root_dir)) - 1)//25
    num_objects = len(os.listdir(os.path.join(root_dir, '0')))




    my_iter = range(num_scenes)
    if process_id == 0:
        my_iter = tqdm(my_iter)

    for scene_idx in my_iter:
        scene_idx *= 25
        scene_idx += process_id
        for object_idx in range(num_objects):
            root_dir = "/mmfs1/home/jrl712/amazon_home/data/isaaclab_sg_5000_300_obj"
            item_path = os.path.join(root_dir, str(scene_idx), f"t_{object_idx}.pkl")
            with open(item_path, 'rb') as f:
                sample = pickle.load(f)

            tmp_save_file = f"/gscratch/scrubbed/test{process_id}.pkl"
            save_dir = "/mmfs1/home/jrl712/amazon_home/data/isaaclab_sg_5000_300_obj_cpu"
            torch.save(sample, tmp_save_file)
            sample2 = torch.load(tmp_save_file, map_location='cpu')

            
            save_dir = os.path.join(save_dir, str(scene_idx))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(sample2, os.path.join(save_dir, f"t_{object_idx}_cpu.pkl"))

if __name__=="__main__":
    with Pool(num_processes) as p:
        p.map(process, list(range(num_processes)))