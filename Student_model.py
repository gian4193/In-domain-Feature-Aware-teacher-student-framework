import torch
from torch import nn
import torchvision.models as models



class Resnet18_get_layer(nn.Module):
    def __init__(self, initial = True, img_size = 32):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        if img_size == 32:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if initial :
            self.resnet.apply(self.init_conv)
        
        
    def init_conv(self,m):
        if isinstance(m, nn.Conv2d):
#             print(self.i,"init weight")
            torch.nn.init.xavier_uniform_(m.weight)
    
                 
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return [layer1,layer2,layer3,layer4]