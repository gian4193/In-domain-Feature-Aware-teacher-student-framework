from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
import copy
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
    plt.figure(figsize=(4,4))
    plt.imshow( (np.transpose(img , (1, 2, 0)) ))
    plt.show()

class Imagedataset(BaseDataset):
    
    def __init__(self, path, transform = None):
        self.PATH = path
        self.img_list = []
        self.transform = transform
        self.image_list()
        
    def image_list(self):
        f = open( self.PATH, 'r')
        for line in f:
            self.img_list.append(line.strip())
        
        
    def __getitem__(self,idx):
        img = cv2.imread(self.img_list[idx])
#         print(self.img_list[idx])
        img = img[:,:,::-1]/255
# #         img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = self.transform(img)
        return img.float()
    
    def __len__(self):
        return len(self.img_list)




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

#     imshow(image_dataset[0])
