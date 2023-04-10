import os
import json
import deepcopy
import collections

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

class FS_VinInpIterator():
    FS_Vint = collections.namedtuple("FS_Vint", ["ncls", "sup_size", "q_size", "unlabeled_size"])

    def __init__(self, ncls, sup_size, q_size, unlabeled_size):
        self.t = self.FS_Vint(ncls, sup_size, q_size, unlabeled_size)

    def inp_instance(self): 
        return deepcopy(self.t)

class EDataset(Dataset):
    def __init__(self, labels, inp_instance_iterator, size, transforms): 
        self.labels = labels
        self.inp_instance_iterator = inp_instance_iterator 
        self.labelset = np.unique(labels)
        self.indices = np.arange(len(labels)) 
        self.transforms = transforms 
        self.reshuffle()
        self.size = size

    def reshuffle(self):
        self.clss_idx = [np.random.permutation(self.indices[self.labels == label])for label in self.labelset]
        self.starts = np.zeros(len(self.clss_idx), dtype=int) 
        self.lengths = np.array([len(x) for x in self.clss_idx])

    def gen_fs_t(self, ncls, size):
        classes = np.random.choice(self.labelset, ncls, replace=False) 
        starts = self.starts[classes]
        reminders = self.lengths[classes] - starts
        if np.min(reminders) < size:
            return None 
        inp_instance_indices = np.array([self.clss_idx[classes[i]][starts[i]:(starts[i] + size)] for i in range(len(classes))])
        inp_instance_indices = np.reshape(inp_instance_indices, [ncls, size]).transpose()
        self.starts[classes] += size
        return inp_instance_indices.flatten()
    
    def inp_instance_t_list(self): 
        t_list = []
        t_info = self.inp_instance_iterator.inp_instance()
        ncls, sup_size, q_size, unlabeled_size = t_info
        unlabeled_size = min(unlabeled_size, self.lengths.min() - sup_size - q_size)
        t_info = FS_VinInpIterator.FS_Vint(ncls=ncls, sup_size=sup_size, q_size=q_size, unlabeled_size=unlabeled_size)
        k = sup_size + q_size + unlabeled_size 
        if np.any(k > self.lengths):
            raise RuntimeError("Requested more inp_instances than existing") 
        fs_t = self.gen_fs_t(ncls, k)

        while fs_t is not None:
            t_list.append((t_info, fs_t))
            t_info = self.inp_instance_iterator.inp_instance() 
            ncls, sup_size, q_size, unlabeled_size = t_info
            k = sup_size + q_size + unlabeled_size
            fs_t = self.gen_fs_t(ncls, k)
        return t_list
    
    def __getitem__(self, idx):
        fs_t_info, indices = self.t_list[idx]
        o_aindices = np.argsort(indices)
        o_indices = np.sort(indices)
        ncls, sup_size, q_size, unlabeled_size = fs_t_info
        k = sup_size + q_size + unlabeled_size
        _images = self.inp_instance_images(o_indices) 
        images = torch.stack([self.transforms(_images[i]) for i in np.argsort(o_aindices)])
        total, c, h, w = images.size()
        images = images.view(k, ncls, c, h, w)
        del(_images)
        images = images * 2 - 1
        targets = np.zeros([ncls * k], dtype=int) 
        targets[o_aindices] = self.labels[o_indices, ...].ravel() 
        inp_instance = {
            "dataset": self.name,
            "channels": c,
            "height": h,
            "width": w,
            "ncls": ncls,
            "sup_size": sup_size,
            "q_size": q_size,
            "unlabeled_size": unlabeled_size, 
            "targets": torch.from_numpy(targets), 
            "sup_set": images[:sup_size, ...], 
            "q_set": images[sup_size:(sup_size + q_size), ...],
            "u_set": None if unlabeled_size == 0 else images[(sup_size + q_size):, ...]
        }
        return inp_instance

    def __iter__(self):
        self.t_list = []
        while len(self.t_list) < self.size:
            self.reshuffle()
            self.t_list += self.inp_instance_t_list()
        return []

class EKannDataset(EDataset): 
    h = 84
    w = 84
    c=3
    split_paths = {"train":"base", "val":"val", "valid":"val", "test":"novel"} 
    
    def __init__(self, data_root, split, input_iterator, size, transforms):
        self.data_root = data_root
        self.split = split
        with open(os.path.join(self.data_root, "fs_lists", "%s.json"%self.split_paths[split]), 'r') as infile: 
            self.metadata = json.load(infile)
        labels = np.array(self.metadata['image_labels'])
        label_map = {l: i for i, l in enumerate(sorted(np.unique(labels)))} 
        labels = np.array([label_map[l] for l in labels]) 
        super().__init__(labels, input_iterator, size, transforms)

    def sample_images(self, indices): 
        return[np.array(Image.open(self.metadata['image_names'][i]).convert("RGB")) for i in indices]
    
    def __iter__(self):
        return super().__iter__()