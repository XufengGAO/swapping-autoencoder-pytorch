"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data 

from PIL import Image
import os
import os.path
import random
import math
from matplotlib import pyplot as plt
import json

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf"), json_path="", load=False):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if load:
        # Opening JSON file
        with open(json_path) as json_file:
             data = json.load(json_file)
             for _, img_path in data.items():
                path = os.path.join(root, fname) 
                images.append(img_path)
    else:
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]
    
def make_nightVision_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    img_dict = {}

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        img_type = root.split("/")[-1]
        img_dict[img_type] = []
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_dict[img_type].append(path)

    
    #train_images, _ = read_split_data(img_dict)
    #images = train_images['day_images'] + train_images['night_images']
    #print('{} day_images and {} night_images'.format(len(train_images['day_images']), len(train_images['night_images'])))

    images = img_dict['day_images'] + img_dict['night_images']

    return images[:min(max_dataset_size, len(images))]

def read_split_data(img_dict: dict, day_rate=50, night_rate=75):
    random.seed(0) # seed to reproduce results

    train_images = {'day_images':[], 'night_images':[]}
    val_images = {}
    val_images['day_images'] = random.sample(img_dict['day_images'], k=int(day_rate))
    val_images['night_images'] = random.sample(img_dict['night_images'], k=int(night_rate))

    json_dict = {}
    for type, imgs in val_images.items():
        for i, img in enumerate(imgs):
            json_dict[type+'_'+str(i)] = img
    # Serializing json
    json_object = json.dumps(json_dict, indent=4)
 
    # Writing to sample.json
    with open("./datasets/nightVisionDatasets/val_images.json", "w") as outfile:
        outfile.write(json_object)


    for type, imgs in img_dict.items():
        for img in imgs:
            if img not in val_images[type]: 
                train_images[type].append(img)

    print('\n-------------Reading Datasets-----------------')
    print("{} images were found in the dataset.".format(len(img_dict['day_images'])+len(img_dict['night_images'])))
    print("{} images for training.".format(len(train_images['day_images'])+len(train_images['night_images'])))
    print("{} images for validation.".format(len(val_images['day_images'])+len(val_images['night_images'])))

    plot_image = True
    image_class = 4
    every_class_num = [len(train_images['day_images']), len(train_images['night_images']), len(val_images['day_images']), len(val_images['night_images'])]
    image_class = ['train_day', 'train_night', 'val_day', 'val_night']
    if plot_image:
        # x-axis, height
        plt.bar(range(len(image_class)), every_class_num, align='center')
        # replace x-axis with class name
        plt.xticks(range(len(image_class)), image_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('image class distribution')
        plt.savefig("./savefigs/nightVision_class_barchart.png")

    return train_images, val_images

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
