

import numpy as np
import torch
import torchvision
from torch import nn, optim, Tensor

import pretrainedmodels



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_conv = pretrainedmodels.se_resnext50_32x4d()
        self.model_conv.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_conv.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        x = self.model_conv(x)
        return x

