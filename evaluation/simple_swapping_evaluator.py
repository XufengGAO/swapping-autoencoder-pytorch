from genericpath import exists
import os
from re import A
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
from matplotlib import pyplot as plt
import util
import torch
import numpy as np

class SimpleSwappingEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_image", required=True, type=str)
        parser.add_argument("--input_texture_image", required=True, type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+',
                            default=[1.0],
                            help="Performs interpolation of the texture image."
                            "If set to 1.0, it performs full swapping."
                            "If set to 0.0, it performs direct reconstruction"
                            )
        
        opt, _ = parser.parse_known_args()
        dataroot = os.path.dirname(opt.input_structure_image)
        
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    def load_RGB_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        return img
    
    def evaluate(self, model, dataset, nsteps=None):
        #images = []
        structure_image = self.load_image(self.opt.input_structure_image)
        #images.append(structure_image.squeeze(0).cpu())
        texture_image = self.load_image(self.opt.input_texture_image)
        

        #print('structure: ', type(structure_image), structure_image.shape)
        #print('texture: ', type(texture_image), texture_image.shape)
        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        
        model(sample_image=structure_image, command="fix_noise")
        structure_code, source_texture_code = model(
            structure_image, command="encode")
        _, target_texture_code = model(texture_image, command="encode")

        alphas = self.opt.texture_mix_alphas
        for alpha in alphas:
            texture_code = util.lerp(
                source_texture_code, target_texture_code, alpha)

            output_image = model(structure_code, texture_code, command="decode")
            #images.append(output_image.squeeze(0).cpu())
            print('output', type(output_image), output_image.shape)
            output_image = transforms.ToPILImage()(
                (output_image[0].cpu().clamp(-1.0, 1.0) + 1.0) * 0.5)

            output_name = "%.2f_%s_%s.png" % (
                alpha,
                os.path.splitext(os.path.basename(self.opt.input_structure_image))[0],
                os.path.splitext(os.path.basename(self.opt.input_texture_image))[0]
            )

            output_path = os.path.join(savedir, output_name)

            output_image.save(output_path)
            print("Saved at " + output_path)

        """
        image = plt.imread(output_path) * 255
        
            print(type(image), image.shape, np.amax(image))

            fig = plt.figure(figsize=(15, 7), dpi=100)
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
            plt.imshow(image.astype('uint8'))
            ax.set_title("%sepo_mixed" % (nsteps))
            fig.savefig("./savefigs/results.jpg")
            print(type(fig))

        images.append(texture_image.squeeze(0).cpu())
        # batching images
        images = torch.stack(images, dim=0)
        print('stack iamges shape', images.shape)
        # width, height
        
        for i in range(len(images)):
            # 1: 子图共1行, num_imgs:子图共num_imgs列, 当前绘制第i+1个子图
            ax = fig.add_subplot(1, len(images), i+1, xticks=[], yticks=[])

            # CHW -> HWC
            npimg = images[i].cpu().numpy().transpose(1, 2, 0)

            # 将图像还原至标准化之前
            # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
            npimg = (npimg * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            plt.imshow(npimg.astype('uint8'), aspect='auto')

            if i == 0:
                title = "input structure image"
            elif i == len(images)-1:
                title = "input texture image"
            elif len(images)>3 and i>0:
                title = "alpla = {}".format(alphas[i-1])
            else:
                title = "output mixed image"
            ax.set_title(title)
            
        fig.savefig("./savefigs/results.jpg")
        """

        return {}
