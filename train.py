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
print('{} unaligned, {} batch and {} real_batch'.format(opt.use_unaligned, opt.batch_size, opt.real_batch_size))

dataset = data.create_dataset(opt)  
dataset_size = len(dataset)
opt.dataset = dataset               

# SummaryWriter instance
if os.path.exists(opt.tb_folder) is False:
    os.makedirs(opt.tb_folder)
tb_writer = SummaryWriter(log_dir=opt.tb_folder)

print('One epoch includes {} batches'.format(len(dataset.dataloader)))  # one epoch 2040 batches

num_epoch = 200
opt.total_nimgs = num_epoch * len(dataset.dataloader) * opt.real_batch_size  # train epochs in total
opt.save_freq = len(dataset.dataloader) * opt.real_batch_size       # save the model per epoch
opt.evaluation_freq = len(dataset.dataloader) * opt.real_batch_size # evaluate the model per epoch
opt.print_freq = 100 
opt.display_freq = 400 

trainOption.print_options(opt)
if trainOption.isTrain:
    trainOption.save_options(opt)

iter_counter = IterationCounter(opt)
visualizer = Visualizer(opt)       
metric_tracker = MetricTracker(opt)
evaluators = GroupEvaluator(opt)

prepare_data = next(dataset) 
model = models.create_model(opt, prepare_data)
optimizer = optimizers.create_optimizer(opt, model)

total_iters = 0
# while not iter_counter.completed_training():
for epoch in range(opt.epoch_count, opt.n_epochs + 1):
    # with iter_counter.time_measurement("data"):
    #     cur_data = next(dataset)
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

    for i, cur_data in enumerate(dataset):
        iter_start_time = time.time()  # timer for computation per iteration

        with iter_counter.time_measurement("train"):
            torch.cuda.empty_cache()
            losses = optimizer.train_one_step(cur_data) # D first, then G, then D again, ...
                                                        # a dict recording different loss values
            metric_tracker.update_metrics(losses, smoothe=True)
        
        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        with iter_counter.time_measurement("maintenance"):
            #if iter_counter.needs_printing():
            if total_iters % opt.print_freq == 0:
                visualizer.print_current_losses(epoch,
                                                epoch_iter,
                                                iter_counter.time_measurements,
                                                metric_tracker.current_metrics())
                for k, v in metric_tracker.current_metrics().items():
                    if k in ['D_mix', 'D_real', 'D_rec', 'PatchD_real', 'PatchD_mix', "D_R1"]:
                        tb_writer.add_scalars('Image_D Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                    if k in ['netF_spatial_loss']:
                        tb_writer.add_scalars('net_F Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                    if k in ['G_GAN_mix', 'G_GAN_rec', 'G_L1', 'G_mix', 'G_spatial_loss']:
                        tb_writer.add_scalars('G Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                    if k in ['D_total', 'G_total']:
                        tb_writer.add_scalars('G and D', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, metric_tracker.current_metrics())
                        
            if total_iters % opt.display_freq == 0:
                visuals = optimizer.get_visuals_for_snapshot(cur_data)
                visualizer.display_current_results(visuals, epoch)
            
            # if iter_counter.needs_evaluation():
            #     metrics = evaluators.evaluate(
            #         model, dataset, iter_counter.steps_so_far)  
            #     metric_tracker.update_metrics(metrics, smoothe=False) 

            iter_counter.record_one_iteration() 
            
    if epoch % opt.save_epoch_freq == 0:         # save the model per epoch
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        optimizer.save(epoch, total_iters)       

            

print('--------------------Training finished--------------------')


"""
# print encoder structure
print("------FromRGB-----")
print(len(model.singlegpu_model.E.FromRGB))
for layer_id, layer in enumerate(model.singlegpu_model.E.FromRGB):
    print(layer_id, layer)

print("------DownToSpatialCode-----")
print(len(model.singlegpu_model.E.DownToSpatialCode))
for layer_id, layer in enumerate(model.singlegpu_model.E.DownToSpatialCode):
    print(layer_id, layer)

print("------ToSpatialCode (sp)-----")
print(len(model.singlegpu_model.E.ToSpatialCode))
for layer_id, layer in enumerate(model.singlegpu_model.E.ToSpatialCode):
    print(layer_id, layer)

print("------DownToGlobalCode-----")
print(len(model.singlegpu_model.E.DownToGlobalCode))
for layer_id, layer in enumerate(model.singlegpu_model.E.DownToGlobalCode):
    print(layer_id, layer)

print("------ToGlobalCode-----")
print(len(model.singlegpu_model.E.ToGlobalCode))
for layer_id, layer in enumerate(model.singlegpu_model.E.ToGlobalCode):
    print(layer_id, layer)
"""

