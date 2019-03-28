import os
import torch
import pandas as pd
import numpy as np
import scipy
import albumentations
from albumentations import torch as AT

from src.model import Model
from src.dataset import CancerDataset


def main(model_dir, data_dir):


    best_train_result_path = os.path.join(model_dir, "best03.pth")

    checkpoint = torch.load(best_train_result_path)
        
    model = Model().cuda()
    model.load_state_dict(checkpoint["model"])

    input = torch.randn(1, 3, 96, 96, device='cuda')

    torch.onnx.export(model, input, './model.onnx')
  

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir

    main(model_dir, data_dir)