"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import util
import os


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():   # check all attributes in file, take the corresponding class
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt): 
    data_loader = ConfigurableDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class DataPrefetcher():
    def __init__(self, dataset):
        self.dataset = dataset
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.dataset)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset)


class ConfigurableDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.initialize(opt.phase)

    def initialize(self, phase):
        opt = self.opt
        self.phase = phase
        if hasattr(self, "dataloader"):
            del self.dataloader
        dataset_class = find_dataset_using_name(opt.dataset_mode)  # return dataset class
        dataset = dataset_class(util.copyconf(opt, phase=phase, isTrain=phase == "train"))  # Set two attributes with specified values with util.copyconf() function
                                                                                            # Then create a dataset_class(opt) instance
        shuffle = phase == "train" if opt.shuffle_dataset is None else opt.shuffle_dataset == "true"  # shuffle
        print("dataset [%s] of size %d was created. shuffled=%s" % (type(dataset).__name__, len(dataset), shuffle))

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.real_batch_size,
            shuffle=shuffle,
            num_workers=int(opt.num_gpus),
            drop_last=phase == "train",
        )
        # The drop_last=True parameter ignores the last batch 
        # (when the number of examples in your dataset is not divisible by 
        # your batch_size ) while drop_last=False will make the last batch smaller than your batch_size

        # self.dataloader = dataset
        self.dataloader_iterator = iter(self.dataloader)  # return iterator
        # self.underlying_dataset = dataset
        self.repeat = phase == "train"  # use data repeatly if training is not finished, see __next()__
        self.length = len(dataset)
        

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
            
    def load_data(self):
        return self

    def __len__(self):
        return self.length

    # def set_phase(self, target_phase):
    #     if self.phase != target_phase:
    #         self.initialize(target_phase)

    # def __iter__(self):  # used with __next__(__iter(self)==self)
    #     self.dataloader_iterator = iter(self.dataloader)
    #     return self

    def __next__(self):
        try:
            return next(self.dataloader_iterator)
        except StopIteration:   # repeat or not
            if self.repeat:
                self.dataloader_iterator = iter(self.dataloader)
                return next(self.dataloader_iterator)
            else:
                raise StopIteration
