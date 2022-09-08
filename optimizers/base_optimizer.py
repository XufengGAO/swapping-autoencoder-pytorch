from models import MultiGPUModelWrapper


class BaseOptimizer():
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt

    def train_one_step(self, data_i, total_steps_so_far=0):
        pass

    def get_visuals_for_snapshot(self, data_i):
        return {}

    def save(self, epoch, total_steps_so_far):
        pass
