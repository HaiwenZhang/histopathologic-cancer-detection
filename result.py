import os
import torch
import pandas as pd
import numpy as np
import scipy
import albumentations
from albumentations import torch as AT

from src.model import Model
from src.dataset import CancerDataset

batch_size = 128
num_workers = 5

def main(model_dir, data_dir):

    data_transforms_test = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.Normalize(),
        AT.ToTensor()
        ])

    data_transforms_tta0 = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.Normalize(),
        AT.ToTensor()
        ])

    data_transforms_tta1 = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.RandomRotate90(p=1),
        albumentations.Normalize(),
        AT.ToTensor()
        ])

    data_transforms_tta2 = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.Transpose(p=1),
        albumentations.Normalize(),
        AT.ToTensor()
        ])

    data_transforms_tta3 = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.Flip(p=1),
        albumentations.Normalize(),
        AT.ToTensor()
        ])

    best_train_result_path = os.path.join(model_dir, "best.pth")
    checkpoint = torch.load(best_train_result_path)

    model = Model()
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    model.eval()
    NUM_TTA = 8

    sigmoid = lambda x: scipy.special.expit(x)
    test_data_dir = os.path.join(data_dir, "test")
    submit_sample_csv = os.path.join(data_dir, "sample_submission.csv")
    result_csv = os.path.join(data_dir, "sub_tta.csv")
    for num_tta in range(NUM_TTA):
        if num_tta==0:
            test_set = CancerDataset(datafolder=test_data_dir, datatype='test', transform=data_transforms_test)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        elif num_tta==1:
            test_set = CancerDataset(datafolder=test_data_dir, datatype='test', transform=data_transforms_tta1)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        elif num_tta==2:
            test_set = CancerDataset(datafolder=test_data_dir, datatype='test', transform=data_transforms_tta2)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        elif num_tta==3:
            test_set = CancerDataset(datafolder=test_data_dir, datatype='test', transform=data_transforms_tta3)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
        else:
            test_set = CancerDataset(datafolder=test_data_dir, datatype='test', transform=data_transforms_tta0)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    
        preds = []
        for batch_i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data).detach()
            pr = output[:,0].cpu().numpy()
            for i in pr:
                preds.append(sigmoid(i)/NUM_TTA)
        if num_tta==0:
            test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
            test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
        else:
            test_preds['preds']+=np.array(preds)
    
    sub = pd.read_csv(submit_sample_csv)
    sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
    sub = sub[['id', 'preds']]
    sub.columns = ['id', 'label']
    sub.to_csv(result_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir

    main(model_dir, data_dir)