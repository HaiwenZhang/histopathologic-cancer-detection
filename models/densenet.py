import torch 
import logging
from torchvision import models
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.pretrained_model = models.densenet169(pretrained=params.use_pretrained)
        self.set_parameter_requires_grad(self.pretrained_model, params.feature_extract)
        num_ftrs = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Linear(num_ftrs, 1)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.sigmoid(x)
        return x

    def set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            logging.info("freeze model parameters")
            for p in model.parameters():
                    p.requires_grad = False
    