# -*- coding: utf-8 -*-
from typing import Dict, Iterable, Callable
from Image_dataset import Imagedataset
import argparse

import torchvision.models as models
import torchsummary as summary
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
import copy
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import timm

import torchvision.transforms.functional as TF
import random


class RotateDataset(BaseDataset):
    def __init__(self, data):
        self.data = data
        self.degree_map = [0, 90, 180, -90]  # default 4 個角度
        self.h_map = [-56,0,56]              # default : 224/4 = 56
        self.v_map = [-56,0,56]
    def __getitem__(self,idx):
        degree = random.randint(0, 3)
        h = random.randint(0, 2)
        v = random.randint(0, 2)
        img = self.data[idx]
#         rotated_img = TF.rotate(img,)
        rotated_img = TF.affine(img, angle=self.degree_map[degree], translate= [self.h_map[h], self.v_map[v]], scale=1.0, shear=0)
        return rotated_img , degree,
    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagespath",
                    help="the file contains all images path ")  
    args = parser.parse_args()
     
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224,224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     #imagenet
    ])
    
    image_dataset = Imagedataset(args.imagespath, transform = transform)
    train_rotated_dataset = RotateDataset(image_dataset)
#     print(train_rotated_dataset[0])

