import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.dir_A = os.path.join(opt.dataroot, 'night_images')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'day_images')  # create a path '/path/to/data/trainB'
        

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "testA")
            self.dir_B = os.path.join(opt.dataroot, "testB")

        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        random.Random(0).shuffle(self.A_paths)
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        random.Random(0).shuffle(self.B_paths)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform = get_transform(opt,  grayscale=False, convert=False)
        if self.opt.isTrain and opt.augment:
            self.transform_aug = transforms.Compose(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                 transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self.transform_aug = None
        self.transform_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.B_indices = list(range(self.B_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if index == 0 and self.opt.isTrain:
            random.shuffle(self.B_indices)
        index_B = self.B_indices[index % self.B_size]
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_pil = self.transform(A_img)
        B_pil = self.transform(B_img)

        # apply image transformation
        A = self.transform_tensor(A_pil)
        B = self.transform_tensor(B_pil)
        if self.opt.isTrain and self.transform_aug is not None:
            A_aug = self.transform_aug(A_pil)
            B_aug = self.transform_aug(B_pil)
            return {'real_A': A, 'real_B': B, 'A_paths': A_path, 'B_paths': B_path, 'aug_A': A_aug, 'aug_B': B_aug}
        else:
            return {'real_A': A, 'real_B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
