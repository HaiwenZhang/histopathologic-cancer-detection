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
from src.utlis import Params

torch.backends.cudnn.benchmark = True

def main(params):
    wd = 4e-4
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(80),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(80),
        transforms.ToTensor(),
        normalize
    ])

    sgd = partial(optim.SGD, lr=params.base_lr, momentum=0.9, weight_decay=wd)

    writer = SummaryWriter(params.model_dir + "/log")
    model = Model(base=models.resnet34).cuda()

    trn_dl, val_dl = get_dateloaders(params, train_transform, valid_transform)

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

    # epoches = 10
    # max_lr = 5e-2
    # warmup_batches = 2 * len(trn_dl)
    # lr1 = np.linspace(max_lr / 25, max_lr, num=warmup_batches, endpoint=False)
    # lr2 = np.linspace(max_lr, max_lr / cfg['rate'], num=epoches * len(trn_dl) - warmup_batches)
    # lrs = np.concatenate((lr1, lr2))
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
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    params.csv_file = os.path.join(args.data_dir, 'train_labels.csv')
    params.data_dir = os.path.join(args.data_dir, 'train')

    params.shuffle_dataset = True
    params.shuffle = True
    params.feature_extract = True
    params.use_pretrained = True
    params.model_dir = args.model_dir

    main(params)

