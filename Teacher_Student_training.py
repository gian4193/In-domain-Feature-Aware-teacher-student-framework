# -*- coding: utf-8 -*-
from typing import Dict, Iterable, Callable
from Image_dataset import Imagedataset
from SSL_dataset import RotateDataset
from Student_model import Resnet18_get_layer
from Feature_extractation import Feature_Extractor_Recursive
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




class Cosine_Sim_Loss(nn.Module):       # thesis equation 3.4
    def __init__(self, feature_list = ['model_layer1','model_layer2','model_layer3','model_layer4'], p=2):
        super(Cosine_Sim_Loss, self).__init__()  
        self.mseloss = nn.MSELoss()
        self.feature_list = feature_list
        self.p = p
    def forward(self,teacher,student):
        mse_loss = 0.0
        for i in range(len(self.feature_list)) :
            s_normalize_tensor = F.normalize(student[i], dim=-1, p=self.p)
            t_normalize_tensor = F.normalize(teacher[self.feature_list[i]], dim=-1, p=self.p)
            mse_loss += self.mseloss(s_normalize_tensor , t_normalize_tensor)
        return mse_loss

class Cosine_Sim_Score(nn.Module):
    def __init__(self, feature_list = ['model_layer1','model_layer2','model_layer3','model_layer4'], p=2):
        super(Cosine_Sim_Score, self).__init__()  
        self.mseloss = nn.MSELoss(reduction='none')
        self.feature_list = feature_list
        self.p = p
    def forward(self,teacher,student):
        distance = torch.tensor([0 for _ in range(student[0].size()[0])]).float()
        for i in range(len(self.feature_list)) :
            s_normalize_tensor = F.normalize(student[i], dim=-1, p=self.p)
            t_normalize_tensor = F.normalize(teacher[self.feature_list[i]], dim=-1, p=self.p)
            recon_err = self.mseloss(s_normalize_tensor, t_normalize_tensor)
            recon_err = recon_err.view(recon_err.size()[0],-1)
            recon_err = torch.sum(recon_err, dim=-1)
            distance += recon_err.detach().cpu()
        return distance.numpy()    

def get_auroc(scores_id, scores_ood)  :
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return  roc_auc_score(labels, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagespath",
                    help = "the file contains all images path") 
    parser.add_argument("--vaildation_indomain_imagespath",
                    help = "the file contains all vaildation indomain images path")
    parser.add_argument("--vaildation_out_of_domain_imagespath",
                    help = "the file contains vaildation out-of-domain images path")
    parser.add_argument("-t", "--teacher_model_weight", default = "Teacher_model.pth", help = "load teacher model weight path" )
    parser.add_argument("-s", "--save_path", default = "Student_model.pth", help = "save path, student model after training" )
    parser.add_argument("-e", "--epoch", type = int, default = 200, help = "training epoches" )
    parser.add_argument("-l", "--learning_rate", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("-b", "--batch_size", type = int, default = 64, help = "batch size")
    args = parser.parse_args()
    
    
    ### Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((224,224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     #imagenet
    ])
    
    image_dataset = Imagedataset(args.imagespath, transform = transform)
    train_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size = args.batch_size,
                                          shuffle=True, num_workers=2)
    
    validation = False
    if args.vaildation_out_of_domain_imagespath != None  and args.vaildation_indomain_imagespath !=None:   ### vaildation_out_of_domain_imagespath & vaildation_indomain_imagespath 都有給才會進行validation
        val_id_image_dataset = Imagedataset(args.vaildation_indomain_imagespath, transform = transform)
        val_id__dataloader = torch.utils.data.DataLoader(val_id_image_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2) 
        val_ood_image_dataset = Imagedataset(args.vaildation_out_of_domain_imagespath, transform = transform)
        val_ood__dataloader = torch.utils.data.DataLoader(val_ood_image_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2)
        validation = True
    
    ### model 
    student_model = Resnet18_get_layer(img_size=224).cuda()
    teacher_model = torch.load(args.teacher_model_weight).cuda()
    FEATURE_LIST = ['model_layer1','model_layer2','model_layer3','model_layer4']
    extract_model = Feature_Extractor_Recursive(model=teacher_model,layers=FEATURE_LIST).cuda()
    
    ### training parameter
    optimizer = optim.Adam(student_model.parameters(), lr = args.learning_rate)
    criterion = Cosine_Sim_Loss(feature_list=FEATURE_LIST)
    ood_eval = Cosine_Sim_Score(feature_list=FEATURE_LIST)
    
    ### training
    auc_record = 0.0
    for epoch in range(args.epoch) :
        print("EPOCH : ",epoch)
        extract_model.eval()
        student_model.train()
        total_loss = 0.0
        
        ### training
        for data in tqdm.tqdm(train_dataloader) :
            img = data.cuda()
            teacher_out = copy.deepcopy(extract_model(img))
            studet_out = student_model(img)
            loss = criterion(teacher = teacher_out, student = studet_out)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("LOSS : ", total_loss/len(train_dataloader))
        
        ### vaildaton      
        if validation :  
            student_model.eval()
            ood_list = []
            id_list = []
            for data in tqdm.tqdm(val_ood__dataloader):
                img = data.cuda()
                with torch.no_grad():
                    student_out = student_model(img)
                    teacher_out = copy.deepcopy(extract_model(img))
                ood_score = ood_eval(teacher = teacher_out, student = student_out)
                ood_list.extend(ood_score)

            for data in tqdm.tqdm(val_id__dataloader):
                img = data.cuda()
                with torch.no_grad():
                    student_out = student_model(img)
                    teacher_out = copy.deepcopy(extract_model(img))
                ood_score = ood_eval(teacher = teacher_out, student = student_out)
                id_list.extend(ood_score)
            auc = get_auroc(ood_list,id_list)
            print("AUC : ",auc) 
            if auc > auc_record:
                auc_record = auc
                torch.save(student_model, args.save_path)
                print("save model.....")
        else :                                       ### 如果沒有valdation 就每次都存
            torch.save(student_model, args.save_path)
            print("save model.....")

        

    

    

    

    

    

