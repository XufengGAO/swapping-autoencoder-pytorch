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

iter_counter = IterationCounter(opt)    # counter instance
visualizer = Visualizer(opt)            # visulaizer instance
metric_tracker = MetricTracker(opt)
evaluators = GroupEvaluator(opt)        # a number of evaluators

model = models.create_model(opt)    # return a multigpu_instance of SAE
optimizer = optimizers.create_optimizer(opt, model) # return a SAE_optimizer instance 

while not iter_counter.completed_training():
    with iter_counter.time_measurement("data"): # time for loading one batch of data
        cur_data = next(dataset)    # one batch of data, dict('real_A'=data, 'path_A'=path_list)

    with iter_counter.time_measurement("train"): # time for training one step
        losses = optimizer.train_one_step(cur_data, iter_counter.steps_so_far)  # D first, then G, then D again, ...
                                                                                # a dict recording different loss values
        metric_tracker.update_metrics(losses, smoothe=True)

    with iter_counter.time_measurement("maintenance"):
        if iter_counter.needs_printing():
            visualizer.print_current_losses(iter_counter.steps_so_far,
                                            iter_counter.time_measurements,
                                            metric_tracker.current_metrics())

        if iter_counter.needs_displaying():
            visuals = optimizer.get_visuals_for_snapshot(cur_data)
            visualizer.display_current_results(visuals,
                                               iter_counter.steps_so_far)

        if iter_counter.needs_evaluation():
            metrics = evaluators.evaluate(
                model, dataset, iter_counter.steps_so_far)  # evaluate the model performance
            metric_tracker.update_metrics(metrics, smoothe=False)

        if iter_counter.needs_saving():
            optimizer.save(iter_counter.steps_so_far)

        if iter_counter.completed_training():
            break

        iter_counter.record_one_iteration()

optimizer.save(iter_counter.steps_so_far)
print('Training finished.')


