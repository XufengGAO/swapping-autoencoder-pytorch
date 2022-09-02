from options import TestOptions
import data
import models
from evaluation import GroupEvaluator

opt = TestOptions().parse()
dataset = data.create_dataset(opt)
opt.dataset = dataset
evaluators = GroupEvaluator(opt)

model = models.create_model(opt)

evaluators.evaluate(model, dataset, opt.resume_iter)
