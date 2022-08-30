# -*- coding: utf-8 -*-

# hook 
# function : extract feature maps

import torch.nn as nn

class Feature_Extractor_Recursive(nn.Module):
    def __init__(self, model: nn.Module =None, layers=None):
        '''
         layers : List, layer name you want to extract
        '''
        super().__init__()
        self.model = model
        self._features = {}
        if self.model == None :
            self.model =  models.vgg16(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()                        
        
        self.layers = layers
        self.unwrap_model(self.model,"model")
                     
    def unwrap_model(self, model ,name):          # 以下程式需依據模型的layer name 規則改寫
        for children_name, children_layer in model.named_children():
            if "_" in children_name:
                children_name = children_name.replace("_", "")
            if isinstance(children_layer, nn.Sequential):
                name = name + "_" + children_name
                self.unwrap_model(children_layer,name)
                name = '_'.join(name.split("_")[:-1])
            elif isinstance(children_layer, nn.Module): 
                if not (isinstance(children_layer, nn.Conv2d) or isinstance(children_layer, nn.MaxPool2d) or isinstance(children_layer, nn.BatchNorm2d) or isinstance(children_layer, nn.ReLU) or isinstance(children_layer, nn.Linear)or isinstance(children_layer, nn.Sigmoid) ) : 
                    name = name + "_" + children_name
                    self.unwrap_model(children_layer,name)
                    name = '_'.join(name.split("_")[:-1])
                    
            children_layer.__name__ = name+"_"+children_name
            if children_layer.__name__ in self.layers:
                children_layer.register_forward_hook(
                                self.save_output()
                            )

    def save_output(self):
        def fn(layer,_,output):
#             print(layer.__name__, " : ",output.size())
            self._features[layer.__name__] = output
        return fn
    
    def forward(self,x):
        _ = self.model(x)
        return self._features
