import os
import numpy as np
import torch
import pandas as pd
import PIL
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class HCDDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_pd = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_pd)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.csv_pd.iloc[index].Id+".tif")
        x = PIL.Image.open(path)
        if self.transform is not None:
            x = self.transform(x)
        y = self.csv_pd.iloc[index].Label
        y = y.astype(np.float32)
        return x, y
    