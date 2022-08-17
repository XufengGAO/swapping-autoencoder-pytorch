import os
from plistlib import load
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import the library
import argparse
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.webp',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--dataroot', type=str, default='./datasets/nightVisionDatesets/')
# Parse the argument
opt, none = parser.parse_known_args()
print(os.path.expanduser(opt.dataroot))
print(opt.dataroot)
assert os.path.isdir(opt.dataroot), '%s is not a valid directory' % dir


print('The directory is', opt.dataroot)
images = []
count = 0
for root, _, fnames in sorted(os.walk(opt.dataroot, followlinks=True)):
    #print(root, len(fnames))
    for fname in fnames:
        if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)

print('total images', len(images))
print('first image path', images[0])


# read the input image
img = Image.open(images[0])  # 

# compute the size(width, height) of image
size = img.size
print("Size of the Original image (w x h):", size)
print("type of image:", type(img))

load_size = 512
crop_size = 256

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img)

print(img_tensor.shape)     # tensor的尺寸是（c, h, w）

# define transformt o resize the image with given size
transform = transforms.Resize(size = (crop_size,load_size), interpolation=transforms.InterpolationMode.BICUBIC)

# apply the transform on the input image
img = transform(img)
print("Size after resize:", img.size)

new_image = img.resize((15, 25))
print("Size after resize:", new_image.size)
# plt.imshow(img)
# plt.show()
