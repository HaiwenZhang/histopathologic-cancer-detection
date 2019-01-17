import os
import numpy as np
import torch
import pandas as pd
import PIL
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T



LABEL_MAP = {
0: "Nucleoplasm",
1: "Nuclear membrane",
2: "Nucleoli",
3: "Nucleoli fibrillar center",   
4: "Nuclear speckles",
5: "Nuclear bodies",
6: "Endoplasmic reticulum",
7: "Golgi apparatus",
8: "Peroxisomes",
9:  "Endosomes",
10: "Lysosomes",
11: "Intermediate filaments", 
12: "Actin filaments",
13: "Focal adhesion sites",
14: "Microtubules",
15: "Microtubule ends",
16: "Cytokinetic bridge",
17: "Mitotic spindle",
18: "Microtubule organizing center",  
19: "Centrosome",
20: "Lipid droplets",
21: "Plasma membrane",
22: "Cell junctions",
23: "Mitochondria",
24: "Aggresome",
25: "Cytosol",
26: "Cytoplasmic bodies",
27: "Rods & rings"}

LABEL_KEYS = LABEL_MAP.keys()

class MultiBandMultiLabelDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
        self.csv_pd = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_pd)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.csv_pd.iloc[index].Id+".png")
        x = PIL.Image.open(path)
        if self.transform:
            x = self.transform(x)
        data = self.csv_pd.iloc[index].Target
        label = list(map(int, data.split(" ")))
        y = self.multilabels_to_vec(28, label)
        return x, y
        

    def multilabels_to_vec(self, vec_len, label_list):
        vec = np.zeros((vec_len,))
        for index in label_list:
            vec[index] = 1.0
        vec = torch.from_numpy(vec)
        return vec.float()


class MultiBandMultiLabelDatasetWithMerge4Channel(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
        self.csv_pd = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_pd)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.csv_pd.iloc[index].Id)
        x = self.__load_multiband_image(path)
        if self.transform:
            x = self.transform(x)
        data = self.csv_pd.iloc[index].Target
        label = list(map(int, data.split(" ")))
        y = self.multilabels_to_vec(28, label)
        return x, y
        
    def __load_multiband_image(self, path):
        image_bands = []
        for brand in self.BANDS_NAMES:
            image_path = path + brand
            image_bands.append(PIL.Image.open(image_path))
        
        image = PIL.Image.merge('RGBA', bands=image_bands)
        image = image.convert("RGB")
        return image

    def multilabels_to_vec(self, vec_len, label_list):
        vec = np.zeros((vec_len,))
        for index in label_list:
            vec[index] = 1.0
        vec = torch.from_numpy(vec)
        return vec.float()

