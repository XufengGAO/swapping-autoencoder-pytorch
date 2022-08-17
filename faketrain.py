import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator


opt = TrainOptions().parse()        # return a Namespace including all required arguments
dataset = data.create_dataset(opt)  # return a ConfigurableDataLoader()
opt.dataset = dataset               # set it as a attribute

cur_data = next(dataset)            # return dict('real_A', 'path_A')
                                    # 'real_A' is a tensor， size = (batch, c, h, w)
                                    # 'path_A' is a tensor, corresponds to sample path, like label
print(type(dataset), type(cur_data), type(dataset.dataloader))
print(cur_data['real_A'].shape, type(cur_data['path_A']))
print(len(cur_data['path_A']))

evaluators = GroupEvaluator(opt)

