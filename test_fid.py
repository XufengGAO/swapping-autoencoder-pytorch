from distutils.cmd import Command
import os
from options import TestOptions
from data import create_dataset
from models import create_model
from evaluation.fid_score import calculate_fid_given_paths
from util.visualizer import save_images
from util import html
import util.util as util
import matplotlib.pyplot as plt
import json
import os
from collections import OrderedDict
import torch
if __name__ == '__main__':
    testOptions = TestOptions()
    opt = testOptions.parse()  # get test options
    # hard-code some parameters for test
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.real_batch_size = 1
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.dataset = dataset
    testOptions.print_options(opt)
    # traverse all epoch for the evaluation
    epoches = []
    for i in range(5,105,5):
        epoches.append(i)
    fid_values = {}
    
    use_texture_file = False
    path = '/home/bozorgta/xugao/gitLocal/swapping-autoencoder-pytorch/day_texture.pt'
    if use_texture_file:
        texture_code = torch.load(path)
    epoches = [100]
    for epoch in sorted(epoches):
        opt.resume_iter = str(epoch)
        model = create_model(opt, None)      # create a model given opt.model and other options
        # create a website
        web_dir = os.path.join(opt.result_dir, opt.name, '{}_{}'.format(opt.phase, opt.resume_iter))  # define the website directory
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.resume_iter))

        with torch.no_grad():
            for i, data_i in enumerate(dataset):
                sp, gl = model(data_i["real_A"].cuda(), command="encode")
                if not use_texture_file:
                    _, texture_code = model(data_i["real_B"].cuda(), command="encode")

                rec_A = model(sp, gl, command="decode")
                fake_B = model(sp, texture_code, command="decode")

                #visuals = {"real_A": data_i["real_A"], "real_B": data_i["real_B"], "rec_A": rec_A, "fake_B": fake_B}
                visuals = {"real_A": data_i["real_A"], "fake_B": fake_B}

                img_path = data_i["A_paths"]     # get image paths
                if i % 100 == 0:  # save images to an HTML file
                    print('processing (%04d)-th image... %s' % (i, img_path))
                save_images(webpage, visuals, img_path, width=int(opt.crop_size))
        paths = [os.path.join(web_dir, 'images', 'fake_B'), os.path.join(web_dir, 'images', 'real_B')]
        fid_value = calculate_fid_given_paths(paths, 50, True, 2048)
        fid_values[int(epoch)] = fid_value
        webpage.save()  # save the HTML
    print(fid_values)

    json_object = json.dumps(fid_values, indent=4)
    # Writing to sample.json
    with open("./results/SAE_unaligned_default_clean/figs/def_use_A_fid.json", "w") as outfile:
        outfile.write(json_object)
    x = []
    y = []
    for key in sorted(fid_values.keys()):
        x.append(key)
        y.append(fid_values[key])
    plt.figure()
    plt.plot(x, y)
    for a, b in zip(x, y):
        plt.text(a, b, str(round(b, 2)))
    plt.xlabel('Epoch')
    plt.ylabel('FID on test set')
    plt.title(opt.name)
    plt.savefig(os.path.join(opt.result_dir, opt.name, 'fid.jpg'))


