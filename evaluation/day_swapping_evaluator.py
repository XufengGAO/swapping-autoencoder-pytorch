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

class DaySwappingEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_image", required=True, type=str)
        
        return parser
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor

    def evaluate(self, model, dataset, nsteps=None):
        structure_image = self.load_image(self.opt.input_structure_image)
        model(sample_image=structure_image, command="fix_noise")
        path = '/home/bozorgta/xugao/gitLocal/swapping-autoencoder-pytorch/day_texture.pt'
        if not os.path.exists(path):
            num = 0
            for i, data_i in enumerate(dataset):
                sp, gl = model(data_i["real_A"].cuda(), command="encode")
                if i == 0:
                    total_gl = torch.zeros_like(gl)
                total_gl += gl
                num += 1
                
            texture_code = total_gl/num
            torch.save(texture_code, path)
            print('Save day texture file')
        else:
            num = 482
            print('Load day texture from file')
            texture_code = torch.load(path)

        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        structure_code, _ = model(structure_image, command="encode")
        output_image = model(structure_code, texture_code, command="decode")

        print('output', type(output_image), output_image[0].shape)

        output_image = transforms.ToPILImage()(
            (output_image[0].cpu().clamp(-1.0, 1.0) + 1.0) * 0.5)
        output_image = output_image.resize((512, 512), Image.ANTIALIAS)
        input_image = transforms.ToPILImage()(
            (structure_image[0].cpu().clamp(-1.0, 1.0) + 1.0) * 0.5)
        input_image = input_image.resize((512, 512), Image.ANTIALIAS)

        fig, ax = plt.subplots(1, 2, figsize=(50,30))
        x, y = 0, 0

        ax[0].imshow(input_image)
        ax[0].set_title("Before", fontsize=60)
        ax[0].axis('off')
        ax[1].imshow(output_image)
        ax[1].set_title("After", fontsize=60)
        ax[1].axis('off')

        fig.tight_layout()
        plt.show()

        output_name = "day_%d_%s.png" % (
            num,
            os.path.splitext(os.path.basename(self.opt.input_structure_image))[0]
        )        
        output_path = os.path.join(savedir, output_name)

        # output_image.save(output_path)
        fig.savefig(output_path)
        print("Saved at " + output_path)

        return {}