import argparse
import pandas as pd
import cv2
import os
import shutil
from fastprogress import progress_bar
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help="Direcotry containning the image data")
# parser.add_argument("--target_dir", help="Direcotry which save propressing img data")

BANDS_NAMES = ['_red.png', '_green.png', '_blue.png', '_yellow.png']

def propressing(csv_file, img_dir, target_dir):
    csv_pd = pd.read_csv(csv_file)
    total = len(csv_pd)
    for index in progress_bar(range(total)):
        name = csv_pd.iloc[index].Id
        path = os.path.join(img_dir, name)
        image_bands = []
        for brand in BANDS_NAMES:
            image_path = path + brand
            image_bands.append(Image.open(image_path))
        image = Image.merge('RGBA', bands=image_bands)
        image = image.convert("RGB")
        new_path = os.path.join(target_dir, name+".png")
        image.save(new_path)

def get_small_data_by_random(data_dir):
    csv_file = os.path.join(data_dir, 'train.csv')
    img_dir = os.path.join(data_dir, 'processing_train')
    target_dir = os.path.join(data_dir, 'small_train')
    small_csv_file = os.path.join(data_dir, "small_data.csv")

    csv_pd = pd.read_csv(csv_file)
    small_data = csv_pd.sample(frac=0.1)
    total = len(small_data)
    for index in progress_bar(range(total)):
        name = csv_pd.iloc[index].Id
        path = os.path.join(img_dir, name+".png")
        new_path = os.path.join(target_dir, name+".png")
        shutil.copy(path, new_path)
        
    small_data.to_csv(small_csv_file)




if __name__ == "__main__":
    args = parser.parse_args()
    # propressing(csv_file, img_dir, target_dir)
    get_small_data_by_random(args.image_dir)