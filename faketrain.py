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

cur_data = next(dataset)            # 返回一个dict('real_A', 'path_A')
                                    # 'real_A'中的是一个tensor， 大小为(batch, c, h, w)
                                    # 'path_A'中的是一个tensor, 对应每一个sample的路经，类似标签
print(type(dataset), type(cur_data), type(dataset.dataloader))
print(cur_data['real_A'].shape, type(cur_data['path_A']))
print(len(cur_data['path_A']))

evaluators = GroupEvaluator(opt)

