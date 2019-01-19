import torch
import numpy as np
from torch.nn import functional as F

def acc(output, target):
    """Computes the accuracy for multiple binary predictions"""
    
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc

