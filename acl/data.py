import os
import pandas as pd
import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from avalanche.benchmarks.classic import SplitMNIST, CORe50, SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet, ImageNet_R, SplitCORe50, SplitCUB200, SplitFood100, SplitNCT, DomainNet
from avalanche.benchmarks.generators import filelist_benchmark, paths_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
import torch.utils.data as data
import random

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_scenario(name, seed, cl_strategy):

        if name == 'SplitCIFAR10':
            if cl_strategy == 'l2p':
                return SplitCIFAR10(n_experiences=5, fixed_class_order=np.arange(10), input_size=(224,224))
            return SplitCIFAR10(n_experiences=5, fixed_class_order=np.arange(10))
        elif name == 'SplitCIFAR100':
            if cl_strategy == 'l2p' or cl_strategy == 'dualprompt' or cl_strategy=='l2p_m' or cl_strategy=='dualprompt_m':
                return SplitCIFAR100(
                        10,
                        return_task_id=False,
                        seed=seed,
                        fixed_class_order=np.arange(100),
                        shuffle=True,
                        class_ids_from_zero_in_each_exp=False,
                        input_size=(224,224)
                    )
            else:
                return SplitCIFAR100(
                        10,
                        return_task_id=False,
                        seed=seed,
                        fixed_class_order=np.arange(100),
                        shuffle=True,
                        class_ids_from_zero_in_each_exp=False,
                    )
        elif name == 'SplitCUB200':
            return SplitCUB200(
                10,
                seed = seed,
                fixed_class_order=np.arange(200),
                shuffle=True,
                class_ids_from_zero_in_each_exp=False,
                return_task_id=False,
            )
        elif name == 'CORe50_nc':
            return CORe50(
                scenario = "nc",
                run=0,
                mini=True,
                object_lvl = False,
            )
        elif name == 'SplitCORe50':
            return SplitCORe50(
                10,
                object_lvl = True,
                mini = True,
                return_task_id=False,
                seed=seed,
                fixed_class_order=np.arange(50),
                shuffle=True,
                class_ids_from_zero_in_each_exp=False,
            )
        
        elif name == 'SplitTinyImageNet':
            if cl_strategy == 'l2p':
                return SplitTinyImageNet(
                    10,
                    return_task_id=False,
                    fixed_class_order=np.arange(200),
                    shuffle=True,
                    class_ids_from_zero_in_each_exp=False,
                    input_size=(224,224)
                )
            else:
                return SplitTinyImageNet(
                    10,
                    return_task_id=False,
                    fixed_class_order=np.arange(200),
                    shuffle=True,
                    class_ids_from_zero_in_each_exp=False,
                )
        elif name == 'DomainNet':
            # data_path = '/data3/jhpark/DomainNet'
            # train_files = [os.path.join(dirpath, filename)
            #    for dirpath, dirnames, filenames in os.walk(data_path)
            #    for filename in filenames
            #    if filename.endswith('train.txt')]
                        
            # test_files = [os.path.join(dirpath, filename)
            #             for dirpath, dirnames, filenames in os.walk(data_path)
            #             for filename in filenames
            #             if filename.endswith('test.txt')]
            
            # transform = torchvision.transforms.Compose(
            #     [
            #         torchvision.transforms.Resize((32,32)),
            #         torchvision.transforms.ToTensor(),
            #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #     ]
            # )

            # return filelist_benchmark(
            #     data_path,
            #     train_files,
            #     test_files,
            #     task_labels = list(range(len(train_files))),
            #     complete_test_set_only=False,
            #     train_transform=transform,
            #     eval_transform=transform,
            #     num_classes=100,
            # )
            return DomainNet('/data3/jhpark/DomainNet', 6)
        elif name == 'SplitMNIST':
            return SplitMNIST(5, fixed_class_order=list(range(0,10)))
        elif name == 'SplitImageNetR':
            return ImageNet_R('/data3/jhpark/imagenet-r',10)
        elif name == 'SplitNCT':
            return SplitNCT('/data3/jhpark/nct', 4)
        elif name == 'SplitFood100':
            return SplitFood100(
                10, input_size=(64,64),
                return_task_id=False,
                seed=seed,
                fixed_class_order=np.arange(100),
                shuffle=True,
                class_ids_from_zero_in_each_exp=False,
            )
        else:
            print('no matching dataset')
            return
        
def get_statistics(dataset):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    
    img_size = {
        "CORe50": [3,32,32],
        "BlurryCIFAR10": [3,32,32],
        "DomainNet": [3,32,32],
        "CIFAR10": [3,32,32],
        "CIFAR100": [3,32,32],
        "TinyImageNet": [3,64,64],
        "ImageNetR": [3,224,224],
        "MNIST": [28,28],
        "CUB200": [32,32],
        "Food100": [64,64],
        "NCT": [3,64,64]
    }

    mean = {
        "CIFAR10": (0.4914, 0.4822, 0.4465),
        "CIFAR100": (0.5071, 0.4865, 0.4409),
        "TinyImageNet": (0.4914, 0.4822, 0.4465),
        "ImageNetR": (0.485, 0.456, 0.406),
        "NCT": (0.485, 0.456, 0.406),
        "DomainNet": (0.5, 0.5, 0.5),
        "CORe50": (0.5, 0.5, 0.5),
        "CUB200": (0.5, 0.5, 0.5),
        "Food100": [64,64],
    }

    std = {
        "CIFAR10": (0.2023, 0.1994, 0.2010),
        "CIFAR100": (0.2673, 0.2564, 0.2762),
        "TinyImageNet":(0.2023, 0.1994, 0.2010),
        "NCT":(0.2023, 0.1994, 0.2010),
        "ImageNetR": (0.229, 0.224, 0.225),
        "DomainNet": (0.5, 0.5, 0.5),
        "CORe50": (0.5, 0.5, 0.5),
        "CUB200": (0.5, 0.5, 0.5),
        "Food100": [64,64],
    }

    
    return {
        'img_size': img_size[dataset], 
        'mean': mean[dataset], 
        'std': std[dataset],
    }

class DeepALDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

class Data:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        self.n_pool = len(X_train)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs.tolist()][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return DeepALDataset(X, Y)

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, DeepALDataset(self.X_train[labeled_idxs.tolist()], self.Y_train[labeled_idxs.tolist()])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        # import pdb;pdb.set_trace()
        return unlabeled_idxs, DeepALDataset(self.X_train[unlabeled_idxs.tolist()], self.Y_train[unlabeled_idxs.tolist()])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), DeepALDataset(self.X_train, self.Y_train)
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs.tolist()], self.Y_train[labeled_idxs.tolist()]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs.tolist()], self.Y_train[unlabeled_idxs.tolist()]