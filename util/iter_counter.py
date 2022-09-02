import os
import numpy as np
import torch
import time


class IterationCounter():
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # actual step = default/batch_size
        parser.add_argument("--total_nimgs", default=25 *
                            (1000 ** 2), type=int)                          
        parser.add_argument("--save_freq", default=50000, type=int)         
        parser.add_argument("--evaluation_freq", default=50000, type=int)   
        parser.add_argument("--print_freq", default=480, type=int)         
        parser.add_argument("--display_freq", default=1600, type=int)       
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.epoch_so_far = 0
        self.real_steps = -1
        self.iter_record_path = os.path.join(
            self.opt.checkpoints_dir, self.opt.name, 'iter.txt')
            
        self.steps_so_far = 0
        if "unaligned" in opt.dataset_mode: 
            #self.batch_size = opt.batch_size * 2
            self.batch_size = opt.real_batch_size
        else:
            self.batch_size = opt.batch_size

        self.time_measurements = {}

        automatically_find_resume_iter = opt.isTrain and opt.continue_train \
            and opt.resume_iter == "latest" and opt.pretrained_name is None

        resume_at_specified_iter = opt.isTrain and opt.continue_train \
            and opt.resume_iter.replace("k", "").isnumeric()  # checks if every character of variable is numeric 

        if automatically_find_resume_iter:  # resume from file
            try:
                self.epoch_so_far, self.steps_so_far, self.real_steps = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
                print('Resuming from iteration %d and real steps at %d' % (self.steps_so_far, self.real_steps))
            except Exception:
                print('Could not load iteration record at %s. '
                      'Starting from beginning.' % self.iter_record_path)
        elif resume_at_specified_iter:  # resume at specific step
            steps = int(opt.resume_iter.replace("epoch", ""))
            if "k" in opt.resume_iter:
                steps *= 1000
            self.steps_so_far = steps
        else:  # from begining
            self.steps_so_far = 0

    def record_one_iteration(self):
        if self.needs_saving():
            np.savetxt(self.iter_record_path,
                       [self.epoch_so_far, self.steps_so_far], delimiter=',', fmt='%d')
            print("Saved current iter count at %s" % self.iter_record_path)
        self.steps_so_far += self.batch_size

    def needs_saving(self):
        return (self.steps_so_far >= self.opt.save_freq) and \
            ((self.steps_so_far % self.opt.save_freq) < self.batch_size) and \
            (self.steps_so_far > self.real_steps)

    def needs_evaluation(self):
        return (self.steps_so_far >= self.opt.evaluation_freq) and \
            ((self.steps_so_far % self.opt.evaluation_freq) < self.batch_size)

    def needs_printing(self):
        return ((self.steps_so_far % self.opt.print_freq) < self.batch_size) and \
            (self.steps_so_far > self.real_steps)

    def needs_displaying(self):
        return (self.steps_so_far % self.opt.display_freq) < self.batch_size

    def completed_training(self):
        return (self.steps_so_far >= self.opt.total_nimgs)  # several epoch finished

    class TimeMeasurement:
        def __init__(self, name, parent):
            self.name = name
            self.parent = parent

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, type, value, traceback):
            torch.cuda.synchronize()
            start_time = self.start_time
            elapsed_time = (time.time() - start_time) / self.parent.batch_size  # time for per sample

            if self.name not in self.parent.time_measurements:
                self.parent.time_measurements[self.name] = elapsed_time
            else:
                prev_time = self.parent.time_measurements[self.name]
                updated_time = prev_time * 0.98 + elapsed_time * 0.02
                self.parent.time_measurements[self.name] = updated_time

    def time_measurement(self, name):
        return IterationCounter.TimeMeasurement(name, self)
