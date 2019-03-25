from torchvision import models
from torch import nn

from tensorboardX import SummaryWriter

from lightai.train import *
import cv2
from torch.utils.data import DataLoader

import shutil
import torch.multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision import transforms
from src.model import Model
from src.dataload import get_dateloaders
from src.metric import BCA
import albumentations
from albumentations import torch as AT
from src.utlis import Params, seed_everything

torch.backends.cudnn.benchmark = True

def main(params):
    wd = 4e-4

    train_transforms = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 
            albumentations.RandomBrightness(), albumentations.RandomContrast(),
            albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
        albumentations.HueSaturationValue(p=0.5), 
        albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    valid_transforms = albumentations.Compose([
        #albumentations.CenterCrop(64, 64),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    sgd = partial(optim.Adam, lr=params.base_lr, momentum=0.9, weight_decay=wd)

    writer = SummaryWriter(params.model_dir + "/log")
    model = Model().cuda()

    trn_dl, val_dl = get_dateloaders(params.data_dir, train_transforms, valid_transforms)

    loss_fn = nn.BCEWithLogitsLoss()
    metric = BCA()
    learner = Learner(model=model, trn_dl=trn_dl, val_dl=val_dl, optim_fn=sgd,
                          metrics=[metric], loss_fn=loss_fn,
                          callbacks=[], writer=writer)
    to_fp16(learner, 512)
    learner.callbacks.append(SaveBestModel(learner, small_better=False, name='best.pth',
                                               model_dir=params.model_dir))

    epoches = params.num_epoches
    warmup_batches = 2 * len(trn_dl)
    lr1 = np.linspace( params.base_lr / 25,  params.base_lr, num=warmup_batches, endpoint=False)
    lr2 = np.linspace( params.base_lr,  params.base_lr / 25, num=epoches * len(trn_dl) - warmup_batches)
    lrs = np.concatenate((lr1, lr2))

    lr_sched = LrScheduler(learner.optimizer, lrs)
    learner.fit(epoches, lr_sched)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # Set the random seed for reproducible experiments
    seed_everything(230)

    params.data_dir = args.data_dir
    params.model_dir = args.model_dir

    main(params)

