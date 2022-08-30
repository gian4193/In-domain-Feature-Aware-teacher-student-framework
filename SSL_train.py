# -*- coding: utf-8 -*-
from typing import Dict, Iterable, Callable
from Image_dataset import Imagedataset
from SSL_dataset import RotateDataset
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

import torch.optim as optim
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagespath",
                    help = "the file contains all images path ")  
    parser.add_argument("-s", "--save_path", default = "Teacher_model.pth", help = "save path, model after SSL training" )
    parser.add_argument("-e", "--epoch", type = int, default = 100, help = "training epoches" )
    parser.add_argument("-l", "--learning_rate", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("-b", "--batch_size", type = int, default = 64, help = "batch size")
    args = parser.parse_args()
     
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224,224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     #imagenet
    ])
    
    ### Dataset
    image_dataset = Imagedataset(args.imagespath, transform = transform)
    train_rotated_dataset = RotateDataset(image_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_rotated_dataset, batch_size = args.batch_size,
                                          shuffle=True, num_workers=2)
    
    ### model 
    
    model = timm.create_model('seresnet34')
    pretrained_model = models.resnet34(pretrained=True)
    model.load_state_dict(pretrained_model.state_dict(),strict=False)
    model.fc = nn.Linear(in_features=512, out_features=4, bias=True)             #如果更改旋轉角度類別數記得改 out_features
    model = model.cuda()
    
    ### training parameter
    largelr_parameters = []   ### only train fc & se_block weights
    for name, _ in model.named_parameters():
        if "se"  in name  or "fc" in name:
            largelr_parameters.append(name)
    optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if n in largelr_parameters],
                        "weight_decay": 5*1e-4,
                        "lr" : args.learning_rate
                    },
                ]
    optimizer = optim.Adam( optimizer_grouped_parameters )
    criterion = nn.CrossEntropyLoss()
    
    ### training 
    acc_record = 0.0
    for epoch in range(args.epoch):
        print("Epoch : ",epoch)
        training_loss = 0.0
        training_acc = 0
        model.train()
        for data in tqdm.tqdm(train_dataloader) :
            img, r_gt= data
            img, r_gt = img.cuda(), r_gt.cuda()
            r_out = model(img)
            loss = criterion(r_out,r_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss
            _, predicted = torch.max(r_out.data, 1)
            training_acc += (predicted == r_gt).sum().item()
        print("LOSS : ", training_loss/len(train_dataloader))
        print("ACC : ", training_acc/len(train_rotated_dataset))
        
        if training_acc > acc_record:
            acc_record = training_acc
            torch.save(model, args.save_path)
            print("save model.....")

    

    
