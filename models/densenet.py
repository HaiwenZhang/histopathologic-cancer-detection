import torch 
import logging
from torchvision import models
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.pretrained_model = models.densenet169(pretrained=params.pretrained)
        self.set_parameter_requires_grad(self.pretrained_model, params.feature_extracting)
        num_ftrs = self.pretrained_model.fc.in_features
        self.output = nn.Sequential(,
            nn.Linear(num_ftrs), 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.output(x)
        return x

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            logging.info("freeze model parameters")
            for p in model.parameters():
                    p.requires_grad = False