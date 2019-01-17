import torch
import numpy as np
from torch.nn import functional as F

def f1_loss(input, target, eps=1E-8):
    """ Micro averaging f1 loss
    .. math::   
        P_{micro}=\frac{\sum_{i=1}^nTP_i}{\sum_{i=1}^nTP_i + \sum_{i=1}^nFP_i} \\
        R_{micro}=\frac{\sum_{i=1}^nTP_i}{\sum_{i=1}^nTP_i + \sum_{i=1}^nFN_i} \\
        F1_{micro}=\frac{2xP_{micro}xR_{micro}}{P_{micro}+R_{micro}}
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(
            target.size(), input.size()))

    input = input.float()
    target = target.float()
    
    tp = torch.sum(input * target, dim=1)
    # TP+FP 就是预测值中所有被预测为C的example数量
    precision = tp / (torch.sum(input, dim=1) + eps)
    # TP + FN 就是正式值中被分类为C的exampl数量
    recall = tp / (torch.sum(target, dim=1) + eps)    
    # 调和平均
    f1 = 2 * precision * recall / (precision + recall + eps)
    # 对所有label的F1 loss取均值
    f1 = torch.mean(f1)    
    return f1


def acc(preds,targs,th=0.5):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()


class FocalLoss(torch.nn.Module):
    """Obejct detection multi labels class loss function
       more https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
    """
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

def test_f1_loss():
    y_pred = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])


    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    py_pred = torch.from_numpy(y_pred)
    py_true = torch.from_numpy(y_true)

    f1 = f1_loss(py_pred, py_true)
    print(f1)

if __name__ == '__main__':
    test_f1_loss()
