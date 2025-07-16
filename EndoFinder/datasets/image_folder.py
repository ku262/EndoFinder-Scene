import os
from torchvision.datasets.folder import is_image_file
from torchvision.datasets.folder import default_loader
import numpy as np
import torch
import itertools
import random
import json
from pathlib import Path
import sys
# Make sure vs is in PYTHONPATH.
base_path = str(Path(__file__).resolve().parent.parent.parent)
if base_path not in sys.path:
    sys.path.append(base_path)

def get_polyp_path(up_path):
    name_list = [os.path.join(up_path, i) for i in os.listdir(up_path) if os.path.isdir(os.path.join(up_path, i))]
    polyp_path = []
    for name in name_list:
        polyp_path.extend([os.path.join(name, "img", i) for i in os.listdir(os.path.join(name, "img"))])

    return polyp_path

def readjsonl(jsonl_path):

    total_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total_data.append(item)
    return total_data

class PolypTwinDataset:
    def __init__(self, transform=None, loader=default_loader):
        random.seed(0)
        query_path = "Polyp-Twin/val_query"
        ref_path = "Polyp-Twin/val_ref"

        self.query_list = [os.path.join(query_path, i)for i in os.listdir(query_path)]
        self.ref_list = [os.path.join(ref_path, i)for i in os.listdir(ref_path)]
        
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):


        query_path = self.query_list[idx]
        ref_path = self.ref_list[idx]
        query_images = self.loader(query_path)
        ref_images = self.loader(ref_path)

        if self.transform:
            query_images = self.transform(query_images)
            ref_images = self.transform(ref_images)

        result = {
            'query': query_images,
            'ref': ref_images,
            'query_path': os.path.basename(query_path)[2:],
            'ref_path': os.path.basename(ref_path)[2:]
        }
        return result

    def __len__(self):

        return len(self.query_list)
    
class MultiPolypJsonlDataset:
    def __init__(self, mode, args, transform=None, loader=default_loader):

        random.seed(0)
        train_total_data = readjsonl(args.train_jsonl_path)
        val_total_data = readjsonl(args.val_jsonl_path)
        if mode == "train":
            print(f"train jsonl path: {args.train_jsonl_path}")
        elif mode == "val":
            print(f"val jsonl path: {args.val_jsonl_path}")
            
        num_list, scene_path, img_paths = [], [], []
        for item in train_total_data:
            num_list.append(item['num'])
            scene_path.append(item['scene_path'][0])
            img_paths.append([i[0] for i in item['img_paths']])

        # print(scene_path)
        num_list, scene_path = np.array(num_list), np.array(scene_path)
        # indices = num_list > input_num
        # img_paths = [img_paths[i] for i in range(len(num_list)) if num_list[i] > input_num]
        # num_list = num_list[indices]
        # scene_path = scene_path[indices]

        sublist_counter = 0
        # 用于存储所有元素及其对应的两个序号列表
        all_elements = []
        sublist_indices = []

        # 遍历img_paths中的每个小列表
        for sublist in img_paths:
            for element in sublist:
                all_elements.append(element)
                sublist_indices.append(sublist_counter)
            sublist_counter += 1
        
        random_indices = np.random.permutation(len(all_elements))
        all_elements = [all_elements[i] for i in random_indices]
        sublist_indices = [sublist_indices[i] for i in random_indices]
        all_elements, sublist_indices = np.array(all_elements), np.array(sublist_indices)

        val_scene_list, query_list, ref_list = [], [], []
        for item in val_total_data:
            val_scene_list.append(item['scene_path'][0])
            query_list.append([i[0] for i in item['query_paths']])
            ref_list.append([i[0] for i in item['ref_paths']])
        
        self.train_imgs = all_elements
        self.train_instance = sublist_indices

        self.query_list = query_list
        self.ref_list = ref_list
        self.val_scene_list = val_scene_list
        self.val_len = len(val_scene_list)

        self.img_paths = img_paths
        self.scene_path = scene_path
        self.loader = loader
        self.transform = transform
        self.mode = mode
        self.boostrap = args.boostrap

    def __getitem__(self, idx: int):

        if self.mode == "train":
            try:
                view_path = self.train_imgs[idx]
                instance_id = self.train_instance[idx]
                path = self.scene_path[instance_id]
                # scene_paths = list(self.img_paths[instance_id]).remove([view_path])
                scene_paths = list(filter(lambda x: x != view_path, self.img_paths[instance_id]))
                if self.boostrap:
                    sampled_scene_paths = random.choices(scene_paths, k=len(scene_paths))
                    scene_paths = sampled_scene_paths
                # scene_paths = random.sample(scene_paths, self.input_num)
                # print(len(scene_paths))
                if self.transform:
                    inputs, targets = [], []
                    for i in scene_paths:
                        inputs.append(self.transform(self.loader(i)))
                    input_images = torch.stack(inputs, dim=0)
                    target_images = self.transform(self.loader(view_path))
                    target_images = target_images.unsqueeze(0)

                result = {
                    'input_images': input_images,  
                    'target_image': target_images,  
                    'instance_id' : instance_id,
                    'path' : path,
                    'img_paths' : [i.split('/')[-1] for i in scene_paths]
                }
                
                return result

            except Exception as e:
                # 记录错误信息并返回占位数据
                print(f"TRAIN Error loading data at index {idx}: {e}")
                return None
                
        
        elif self.mode == "val":
            try:
                query_paths = self.query_list[idx]
                ref_paths = self.ref_list[idx]

                query_path = self.val_scene_list[idx]
                ref_path = self.val_scene_list[idx]
                
                # query_paths = random.sample(query_paths, 1)
                # ref_paths = random.sample(ref_paths, 1)
                # print(f"query: {query_paths}")
                # print(f"ref: {ref_paths}")
                # print(f"scene: {query_path}")

                query_images = [self.loader(i) for i in query_paths]
                ref_images = [self.loader(i) for i in ref_paths]

                if self.transform:
                    query, ref = [], []
                    for i in query_images:
                        query.append(self.transform(i))
                    query_images = torch.stack(query, dim=0)
                    for i in ref_images:
                        ref.append(self.transform(i))
                    ref_images = torch.stack(ref, dim=0)

                result = {
                    'query': query_images,
                    'ref': ref_images,
                    'query_path': query_path,
                    'ref_path': ref_path,
                    'query_img_paths': [i.split('/')[-1] for i in query_paths],
                    'ref_img_paths': [i.split('/')[-1] for i in ref_paths]
                }
                return result
            
            except Exception as e:
                # 记录错误信息并返回占位数据
                print(f"VAL Error loading data at index {idx}: {e}")
                return None

    def __len__(self):
        if self.mode == "train":
            return len(self.train_imgs)
        elif self.mode == "val":
            return self.val_len

class MultiImagesPolypJsonlDataset:
    def __init__(self, args, transform=None, loader=default_loader):

        train_total_data = readjsonl(args.train_jsonl_path)
            
        num_list, scene_path, img_paths, ground_truth = [], [], [], []
        for item in train_total_data:
            num_list.append(item['num'])
            scene_path.append(item['scene_path'])
            img_paths.append([i[0] for i in item['img_paths']])
            ground_truth.append(item['ground_truth'])

        num_list, scene_path, ground_truth = np.array(num_list), np.array(scene_path), np.array(ground_truth)
        
    
        self.img_paths = img_paths
        self.scene_path = scene_path
        self.ground_truth = ground_truth
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):

        paths = self.img_paths[idx]

        scene_path = self.scene_path[idx]

        images = [self.loader(i) for i in paths]

        if self.transform:
            trans_images = []
            for i in images:
                trans_images.append(self.transform(i))
            trans_images = torch.stack(trans_images, dim=0)

        result = {
            'images': trans_images,
            'scene_path': scene_path,
            'ground_truth': self.ground_truth[idx],
            'img_paths': [i.split('/')[-1] for i in paths]
        }
        return result

    def __len__(self):
        
        return len(self.scene_path)

class MultiViewPolypJsonlDataset:
    def __init__(self, args, mode, transform=None, loader=default_loader):

        random.seed(0)
        total_data = readjsonl(args.total_jsonl_path)
            
        num_list, scene_path, img_paths, ground_truth = [], [], [], []
        if mode == "val":
            for item in total_data:
                num_list.append(item['num'])
                scene_path.append(item['scene_path'][0])
                total_paths = [i[0] for i in item['query_paths']]
                total_paths.extend([i[0] for i in item['ref_paths']])
                img_paths.append(total_paths)
                ground_truth.append(item['ground_truth'])
        elif mode == "train":
            for item in total_data:
                num_list.append(item['num'])
                scene_path.append(item['scene_path'][0])
                img_paths.append([i[0] for i in item['img_paths']])
                ground_truth.append(item['ground_truth'])

        # print(scene_path)
        num_list, scene_path, ground_truth = np.array(num_list), np.array(scene_path), np.array(ground_truth)
    
        self.img_paths = img_paths
        self.scene_path = scene_path
        self.ground_truth = ground_truth
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):

        paths = self.img_paths[idx]

        scene_path = self.scene_path[idx]

        # paths = random.choices(paths, k=3)
        # paths = random.sample(paths, k=3)

        images = [self.loader(i) for i in paths]

        if self.transform:
            trans_images = []
            for i in images:
                trans_images.append(self.transform(i))
            trans_images = torch.stack(trans_images, dim=0)

        result = {
            'images': trans_images,
            'scene_path': scene_path,
            'ground_truth': self.ground_truth[idx],
            'img_paths': [i.split('/')[-1] for i in paths]
        }
        return result

    def __len__(self):
        
        return len(self.scene_path)

class TestPolypJsonlDataset:
    def __init__(self, args, transform=None, loader=default_loader):

        random.seed(0)
        train_total_data = readjsonl(args.train_jsonl_path)
        val_total_data = readjsonl(args.val_jsonl_path)
            
        num_list, scene_path, query_list, ref_list, ground_truth = [], [], [], [], []
        for item in val_total_data:
            num_list.append(item['num'])
            scene_path.append(item['scene_path'][0])
            total_paths = [i[0] for i in item['query_paths']]
            query_list.append(total_paths)
            total_paths = [i[0] for i in item['ref_paths']]
            ref_list.append(total_paths)
            ground_truth.append(item['ground_truth'])
        for item in train_total_data:
            num_list.append(item['num'])
            scene_path.append(item['scene_path'][0])
            total_paths = [i[0] for i in item['img_paths']]
            query_list.append(total_paths[:2])
            ref_list.append(total_paths[2:])
            ground_truth.append(item['ground_truth'])

        # print(scene_path)
        num_list, scene_path, ground_truth = np.array(num_list), np.array(scene_path), np.array(ground_truth)
    
        self.query_list = query_list
        self.ref_list = ref_list
        self.scene_path = scene_path
        self.ground_truth = ground_truth
        self.loader = loader
        self.transform = transform

    def __getitem__(self, idx: int):

        query_paths = self.query_list[idx]
        ref_paths = self.ref_list[idx]

        query_path = self.scene_path[idx]
        ref_path = self.scene_path[idx]
        
        # query_paths = random.sample(query_paths, 1)
        # ref_paths = random.sample(ref_paths, 1)
        # print(f"query: {query_paths}")
        # print(f"ref: {ref_paths}")
        # print(f"scene: {query_path}")

        query_images = [self.loader(i) for i in query_paths]
        ref_images = [self.loader(i) for i in ref_paths]

        if self.transform:
            query, ref = [], []
            for i in query_images:
                query.append(self.transform(i))
            query_images = torch.stack(query, dim=0)
            for i in ref_images:
                ref.append(self.transform(i))
            ref_images = torch.stack(ref, dim=0)

        result = {
            'query': query_images,
            'ref': ref_images,
            'query_path': query_path,
            'ref_path': ref_path,
            'query_img_paths': [i.split('/')[-1] for i in query_paths],
            'ref_img_paths': [i.split('/')[-1] for i in ref_paths]
        }
        return result


    def __len__(self):
        
        return len(self.scene_path)

if __name__ == "__main__":
    print(__name__)