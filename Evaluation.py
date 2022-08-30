# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from typing import Dict, Iterable, Callable
from Image_dataset import Imagedataset
from SSL_dataset import RotateDataset
from Student_model import Resnet18_get_layer
from Feature_extractation import Feature_Extractor_Recursive
from Teacher_Student_training import Cosine_Sim_Score, get_auroc
import argparse

import torchvision.models as models
import torchsummary as summary
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
import copy
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import timm

import torchvision.transforms.functional as TF
import random

import torch.optim as optim
import tqdm

from sklearn.metrics import roc_auc_score
import numpy as np
import cv2

### 畫 feature maps
def ts_subplot_feature_map(npimg, teacher_feature_maps, student__feature_maps, t_feature_list, s_feature_list, x=2, y=5):
    fig , ax = plt.subplots(x, y, sharex=True, sharey=True,figsize=(15,10))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    plt.subplot(x, y, 1)
    npimg = np.transpose(npimg,(1,2,0))
    plt.imshow(npimg)
    plt.subplot(x, y, y+1)
#     npimg = np.transpose(npimg,(1,2,0))
    plt.imshow(npimg)
    for idx, feature in enumerate(t_feature_list):
        heatmap = teacher_feature_maps[feature].pow(2).mean(1).detach().cpu().numpy()
        heatmap = np.transpose(heatmap,(1,2,0))
#         print(heatmap.shape)
        heatmap = np.maximum(heatmap, 0)   #ReLU
        heatmap /= np.max(heatmap) #正則化
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        
        plt.subplot(x, y, idx+2)
        plt.imshow(npimg, alpha=0.6)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        
        
        heatmap = student__feature_maps[s_feature_list[idx]].pow(2).mean(1).detach().cpu().numpy()
        heatmap = np.transpose(heatmap,(1,2,0))
#         print(heatmap.shape)
        heatmap = np.maximum(heatmap, 0)   #ReLU
        heatmap /= np.max(heatmap) #正則化
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        
        plt.subplot(x, y, idx+2+y)
        plt.imshow(npimg, alpha=0.6)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
#         plt.title(''.join(feature.split("_")[2:]))
    plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--indomain_imagespath",
                    help = "the file contains all indomain images path")
    parser.add_argument("--out_of_domain_imagespath",
                    help = "the file contains  out-of-domain images path")
    parser.add_argument("-t", "--teacher_model_weight", default = "Teacher_model.pth", help = "load teacher model weight path" )
    parser.add_argument("-s", "--student_model_weight", default = "Student_model.pth", help = "load student model weight path" )
    parser.add_argument("-v", "--visualiztion", type = bool , default = False, help = "visualize teacher and student model feature maps")
    args = parser.parse_args()
    
    
    
    ### dataset 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224,224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     #imagenet
    ])
    id_image_dataset = Imagedataset(args.indomain_imagespath, transform = transform)
    id_dataloader = torch.utils.data.DataLoader(id_image_dataset, batch_size = 64, shuffle=False, num_workers=2) 
    ood_image_dataset = Imagedataset(args.out_of_domain_imagespath, transform = transform)
    ood_dataloader = torch.utils.data.DataLoader(ood_image_dataset, batch_size = 64, shuffle=False, num_workers=2)
    
    
    ### model
    teacher_model = torch.load(args.teacher_model_weight).cuda()
    T_FEATURE_LIST = ['model_layer1','model_layer2','model_layer3','model_layer4']
    teacher_extract_model = Feature_Extractor_Recursive(model=teacher_model,layers=T_FEATURE_LIST).cuda()

    student_model = torch.load(args.student_model_weight)
    S_FEATURE_LIST = ['model_resnet_layer1','model_resnet_layer2','model_resnet_layer3','model_resnet_layer4']
    student_extract_model = Feature_Extractor_Recursive(model=student_model,layers=S_FEATURE_LIST).cuda()
    
    
    ### eval
    ood_eval = Cosine_Sim_Score(feature_list=T_FEATURE_LIST)
    teacher_model.eval()
    student_model.eval()
    ood_list = []
    id_list = []
    
    for data in tqdm.tqdm(ood_dataloader):
        img = data.cuda()
        with torch.no_grad():
            student_out = student_model(img)
            teacher_out = copy.deepcopy(teacher_extract_model(img))
        ood_score = ood_eval(teacher = teacher_out, student = student_out)
        ood_list.extend(ood_score)

    for data in tqdm.tqdm(id_dataloader):
        img = data.cuda()
        with torch.no_grad():
            student_out = student_model(img)
            teacher_out = copy.deepcopy(teacher_extract_model(img))
        ood_score = ood_eval(teacher = teacher_out, student = student_out)
        id_list.extend(ood_score)
        
    print("AUC: ",get_auroc(ood_list,id_list))
    
    
    if args.visualiztion :
        print("In-domain")
        for data in tqdm.tqdm(id_image_dataset):
            img = data.unsqueeze(0).cuda()
            t_feature_map = copy.deepcopy(teacher_extract_model(img))
            s_feature_map = copy.deepcopy(student_extract_model(img))
            ts_subplot_feature_map(np.array(data), t_feature_map, s_feature_map, T_FEATURE_LIST, S_FEATURE_LIST)
         
        print("Out-of-domain")
        for data in tqdm.tqdm(ood_image_dataset):
            img = data.unsqueeze(0).cuda()
            t_feature_map = copy.deepcopy(teacher_extract_model(img))
            s_feature_map = copy.deepcopy(student_extract_model(img))
            ts_subplot_feature_map(np.array(data), t_feature_map, s_feature_map, T_FEATURE_LIST, S_FEATURE_LIST)

