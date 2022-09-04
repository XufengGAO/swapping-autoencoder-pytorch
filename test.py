from options import TestOptions
import data
import models
from evaluation import GroupEvaluator

opt = TestOptions().parse()
dataset = data.create_dataset(opt)
opt.dataset = dataset
evaluators = GroupEvaluator(opt)

model = models.create_model(opt)

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

evaluators.evaluate(model, dataset, opt.resume_iter)
