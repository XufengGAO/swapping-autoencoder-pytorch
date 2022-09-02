import imp
import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import numpy as np
from data.image_folder import make_nightVision_dataset
from PIL import Image

class nightVisionDataset(BaseDataset):
    # Import datasets
    def __init__(self, opt):
        self.opt = opt

        # Get paths for all dataset images
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_nightVision_dataset(self.dir_A))
        self.A_size = len(self.A_paths)

        # Get transforms.Compose()
        self.transform_A = get_transform(self.opt, grayscale=False)


    # get a data point from data loader.
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        return self.getitem_by_path(A_path)

    def getitem_by_path(self, A_path):
        try:
            A_img = Image.open(A_path).convert('RGB')
        except OSError as err:
            print('getitem_by_path error', err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        # apply image transformation
        A = self.transform_A(A_img)

        return {'real_A': A, 'path_A': A_path}

    # Size of dataset
    def __len__(self):
        return self.A_size
    
