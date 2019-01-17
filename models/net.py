import logging
import torch.nn as nn
from torchvision import models

def initialize_pretrained_model(model_name, num_labels, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    elif model_name == "res":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    elif model_name == "desenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_labels)
        input_size = 224
    return model_ft, input_size

class MyModel(nn.Module):
    def __init__(self, params):
        super(MyModel, self).__init__()        
        self.pretrained_model, self.input_size =initialize_pretrained_model(params.model_name,
                                                                            params.num_labels,
                                                                            params.feature_extract,
                                                                            params.use_pretrained)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_model(x)
        # x = self.sigmoid(x)
        return x

    def image_size(self):
        return self.input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        logging.info("freeze model parameters")
        for param in model.parameters():
            param.requires_grad = False



class Resnet34_4(nn.Module):
    def __init__(self, pre=True):
        super().__init__()
        encoder = resnet34(pretrained=pre)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if(pre):
            w = encoder.conv1.weight
            self.conv1.weight = nn.Parameter(torch.cat((w,
                                    0.5*(w[:,:1,:,:]+w[:,2:,:,:])),dim=1))
        
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1,self.relu,self.bn1,self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        #the head will be added automatically by fast.ai
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x