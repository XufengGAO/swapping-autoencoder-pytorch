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

print('---------------------Training Start----------------------')
print('Check GPU first:', torch.cuda.is_available(), torch.cuda.device_count())
print('Start Tensorboard with with "tensorboard --logdir ./runs/xxx/ --samples_per_plugin=images=30", view at http://localhost:6006/')


trainOption = TrainOptions()
opt = trainOption.parse()
if opt.use_unaligned:
    opt.real_batch_size = opt.batch_size // 2
else:
    opt.real_batch_size = opt.batch_size
print('{} unaligned, {} batch and {} real_batch'.format(opt.use_unaligned, opt.batch_size, opt.real_batch_size))

dataset = data.create_dataset(opt)  
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
opt.print_freq = 240   
opt.display_freq = 800 

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

print("-------ToStructureModel-------")
print(len(model.singlegpu_model.E.ToStructureModel))
for layer_id, layer in enumerate(model.singlegpu_model.E.ToStructureModel):
    print(layer_id, layer)


while not iter_counter.completed_training():
    with iter_counter.time_measurement("data"):
        cur_data = next(dataset)                # one batch of data, dict('real_A'=data, 'path_A'=path_list)

    with iter_counter.time_measurement("train"):
        torch.cuda.empty_cache()
        losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)  # D first, then G, then D again, ...
                                                                                # a dict recording different loss values
        metric_tracker.update_metrics(losses, smoothe=True)
    
    with iter_counter.time_measurement("maintenance"):
        if iter_counter.needs_printing():
            visualizer.print_current_losses(iter_counter.steps_so_far,
                                            iter_counter.time_measurements,
                                            metric_tracker.current_metrics())
            for k, v in metric_tracker.current_metrics().items():
                if k in ['D_mix', 'D_real', 'D_rec', 'PatchD_real', 'PatchD_mix']:
                    tb_writer.add_scalars('Image_D Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                if k in ['PatchD_real', 'PatchD_mix']:
                    tb_writer.add_scalars('Patch_D Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                if k in ['G_GAN_mix', 'G_GAN_rec', 'G_L1', 'G_mix']:
                    tb_writer.add_scalars('G Loss', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                if k in ['D_total', 'G_total']:
                    tb_writer.add_scalars('G and D', {k:float(format(v.mean(), '.3f'))}, (iter_counter.steps_so_far//opt.print_freq))
                    
        if iter_counter.needs_displaying():
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals,
                                               iter_counter.steps_so_far)
        
        if iter_counter.needs_evaluation():
            metrics = evaluators.evaluate(
                model, dataset, iter_counter.steps_so_far)  
            metric_tracker.update_metrics(metrics, smoothe=False)  
        

        if iter_counter.needs_saving():         # save the model per epoch
            iter_counter.epoch_so_far += 1
            print("Saved model at {} epo and {} steps".format(iter_counter.epoch_so_far, iter_counter.steps_so_far))
            optimizer.save(iter_counter.epoch_so_far, iter_counter.steps_so_far)       
                

        if iter_counter.completed_training():
            break

        iter_counter.record_one_iteration()


iter_counter.epoch_so_far += 1
np.savetxt(iter_counter.iter_record_path,
            [iter_counter.epoch_so_far, iter_counter.steps_so_far], delimiter=',', fmt='%d')
print("End, Saved model at {} epo and {} steps".format(iter_counter.epoch_so_far, iter_counter.steps_so_far))
optimizer.save(iter_counter.epoch_so_far, iter_counter.steps_so_far) # save the model

"""
print('--------------------Training finished--------------------')


