from torchvision import models
from torch import nn

from tensorboardX import SummaryWriter

from lightai.train import *
import cv2
from torch.utils.data import DataLoader
from src.model import Model
import shutil
import torch.multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
torch.backends.cudnn.benchmark = True


def main(params):
    print(params)
    sz = 512


    df = pd.read_csv('../data/full.csv')
    wd = 4e-4
    sgd = partial(optim.SGD, lr=0, momentum=0.9, weight_decay=wd)



    writer = SummaryWriter(f'./log/{name}')


    model = Model(base=models.resnet34).cuda()

    loss_fn = nn.BCELoss()
    metric = None
    learner = Learner(model=model, trn_dl=trn_dl, val_dl=val_dl, optim_fn=sgd,
                          metrics=[metric], loss_fn=loss_fn,
                          callbacks=[], writer=writer)
    to_fp16(learner, 512)
    learner.callbacks.append(SaveBestModel(learner, small_better=False, name='best.pkl',
                                               model_dir=params.model_dir))

    epoches = 10
    warmup_batches = 2 * len(trn_dl)
    lr1 = np.linspace( cfg['base_lr'] / 25,  cfg['base_lr'], num=warmup_batches, endpoint=False)
    lr2 = np.linspace( cfg['base_lr'],  cfg['base_lr'] / 25, num=epoches * len(trn_dl) - warmup_batches)
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

    cfg = {}
    cfg['folds'] = [int(fold) for fold in args.fold]
    cfg['bs'] = 96
    cfg['model_name'] = 'res18_shisu'
    cfg['base_lr'] = 5e-2
    num_workers = 6
    main(cfg)

