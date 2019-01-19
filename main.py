import os
import argparse
import logging
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms

from data.dataload import get_dateloaders
from models.densenet import Net
from train import train_and_evaluate
from torch.optim import lr_scheduler

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help="Direcotry containning the image data")
parser.add_argument("--model_dir", help="Directory containning params.json")

shuffle_dataset = True
shuffle = True
feature_extract = True
use_pretrained = True
image_size = 224

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def setup_and_train(parmas):
    model = Net(params).cuda() if params.cuda else Net(params)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])

    loss_fn = nn.BCELoss()

    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.SGD([
    #            {'params': model.base_parameters},
    #            {'params': model.last_parameters, 'lr': 1e-2}
    #        ], lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gama)

    dataloaders = get_dateloaders(params,
                                  train_transform=train_transform,
                                  valid_transform=valid_transform)

    train_and_evaluate(model=model,
                       dataloaders=dataloaders,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       scheduler=exp_lr_scheduler,
                       params=params)


if __name__ == "__main__":
    # Load the parameters from json file
    #args = parser.parse_args()
    #model_dir = args.model_dir
    #data_dir = args.image_dir
    data_dir = "/home/haiwen/kaggle/histopathologic-cancer-detection"
    model_dir = "./experiments/base_model"
    json_path = os.path.join(model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    params.csv_file = os.path.join(data_dir, 'train_labels.csv')
    params.data_dir = os.path.join(data_dir, 'train')
    params.model_metrics_file = os.path.join(model_dir, "metrics.csv")
    params.shuffle_dataset = shuffle_dataset
    params.shuffle = shuffle
    params.feature_extract = feature_extract
    params.use_pretrained = use_pretrained
    params.model_dir = model_dir

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    setup_and_train(params)
