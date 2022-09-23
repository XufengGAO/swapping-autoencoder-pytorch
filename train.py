import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator
import torch
from torch.utils.cpp_extension import CUDA_HOME;
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time

print('---------------------Training Start----------------------')
print('Check GPU first:', torch.cuda.is_available(), torch.cuda.device_count())
#print('Start Tensorboard with with "tensorboard --logdir ./runs/xxx/ --samples_per_plugin=images=30", view at http://localhost:6006/')

trainOption = TrainOptions()
opt = trainOption.parse()
if opt.use_unaligned:
    opt.real_batch_size = opt.batch_size // 2
else:
    opt.real_batch_size = opt.batch_size
print('{} unaligned, {} batch size and {} real_batch size'.format(opt.use_unaligned, opt.batch_size, opt.real_batch_size))

dataset = data.create_dataset(opt)  
dataset_size = len(dataset)
opt.dataset = dataset               

print('One epoch includes {} batches'.format(len(dataset.dataloader)))

num_epoch = 200
opt.print_freq = 5 * opt.batch_size
opt.display_freq = 10 * opt.batch_size

trainOption.print_options(opt)
if trainOption.isTrain:
    trainOption.save_options(opt)

visualizer = Visualizer(opt)       
metric_tracker = MetricTracker(opt)

if opt.use_NCE:
    prepare_data = next(dataset)
else:
    prepare_data = None 

model = models.create_model(opt, prepare_data)
optimizer = optimizers.create_optimizer(opt, model)

total_iters = 0

for epoch in range(opt.epoch_count, opt.n_epochs + 1):
    epoch_iter = 0                
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    for i, cur_data in enumerate(dataset):
        torch.cuda.empty_cache()
        losses = optimizer.train_one_step(cur_data) # D first, then G, then D again, ...
        metric_tracker.update_metrics(losses, smoothe=True)
        
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        if total_iters % opt.print_freq == 0:   # plot losses
            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, metric_tracker.current_metrics())
                    
        if total_iters % opt.display_freq == 0: # display visudals
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals, epoch)

        if total_iters % opt.save_latest_freq == 0: # cache our latest model every <save_latest_freq> iterations
            print('fast model (epoch %d, total_iters %d)' % (epoch, total_iters))
            optimizer.save(epoch, total_iters)

            
    if epoch % opt.save_epoch_freq == 0:    # save the model per epoch
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        optimizer.save(epoch, total_iters)       

print('--------------------Training finished--------------------')