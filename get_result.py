import pdb
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.utlis import Params
from src.model import Model
from src.dataset import HCDDataset
torch.backends.cudnn.benchmark = True

def main(params):
    submission = pd.read_csv(params.csv_file)
    normalize = transforms.Normalize(mean=[0.70017236, 0.5436771, 0.6961061], std=[0.22246036, 0.26757348, 0.19798167])  
    test_transform = torchvision.transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize])

    test_dataset = HCDDataset(csv_file=params.csv_file, root_dir=params.data_dir, transform=test_transform)
    
    test_idx = list(range(len(submission)))
    test_sampler = SubsetRandomSampler(test_idx)
    test_dl = DataLoader(test_dataset, batch_size=params.batch_size, sampler=test_sampler, num_workers=params.num_workers, pin_memory=True)

    model = Model(base=torchvision.models.resnet34)
    model.half()
    res = []
    with torch.no_grad():
        checkpoint = torch.load(params.model_dir + "/best.pth")
        model.load_state_dict(checkpoint["model"])
        model.cuda()
        model.eval()
        predicts = []
        for idx, imgs in tqdm(zip(test_sampler, test_dl), total=len(test_dl)):
            imgs = imgs[0]
            pred = model(imgs.cuda()).sigmoid().cpu().numpy()
            print(pred)
            #predicts.append(pred)
    #res.append(np.concatenate(predicts,axis=0))

    #np.save(os.path.join(params.data_dir, "submit"),res)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir

    #data_dir = '/home/haiwen/kaggle/histopathologic-cancer-detection'
    #model_dir = './experiments'

    json_path = os.path.join(model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.model_dir = model_dir
    params.csv_file = os.path.join(data_dir, 'sample_submission.csv')
    params.data_dir = os.path.join(data_dir, 'test')
    main(params)
