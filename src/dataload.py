import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import CancerDataset


def get_train_and_valid_dataload(base_dir,
  train_transform,
  valid_transform,
  batch_size = 64,
  num_workers = 5,
  random_seed = 230):

  labels = pd.read_csv(os.path.join(base_dir, 'train_labels.csv'))

  tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.15, random_state=random_seed)

  img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}

  train_set = CancerDataset(datafolder=os.path.join(base_dir, 'train'), datatype='train', transform=train_transform, labels_dict=img_class_dict)
  valid_set = CancerDataset(datafolder=os.path.join(base_dir, 'train'), datatype='train', transform=valid_transform, labels_dict=img_class_dict)

  train_sampler = SubsetRandomSampler(list(tr.index))
  valid_sampler = SubsetRandomSampler(list(val.index))

  # prepare data loaders (combine dataset and sampler)
  train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
  valid_loader = DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

  return train_loader, valid_loader



def get_test_dataloader(base_dir,
  test_transform,
  batch_size = 64,
  num_workers = 2,
  random_seed = 230):
  test_set = CancerDataset(datafolder=os.path.join(base_dir, 'test/'), datatype='test', transform=test_transform)
  test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
  return test_loader