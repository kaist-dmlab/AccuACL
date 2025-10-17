
from typing import Optional, TYPE_CHECKING

import numpy as np
from torch.utils.data.dataloader import DataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate
import torch
import random
import os
from torchvision import datasets
from PIL import Image
import torch.utils.data as data
from avalanche.benchmarks.utils.flat_data import FlatData

    
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

class DeepALPlugin(SupervisedPlugin):

    def __init__(
        self,
        al_strategy,
        fraction = 0.2,
        batch_size = 1,
        model = None,
    ):
        super().__init__()
        self.fraction = fraction
        self.al_strategy = al_strategy
        self.model = model
        self.batch_size = batch_size
        self.selection_result = None

    def before_train_dataset_adaptation(
        self,
        strategy: "SupervisedTemplate",
        custom_selection_result = None,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """

        assert strategy.experience.dataset is not None

        print(len(strategy.experience.dataset))
        X = torch.stack([data[0] for data in strategy.experience.dataset])
        y = strategy.experience.dataset.targets
        data = Data(X, y)
        al_strategy = self.al_strategy(data, net=self.model, mem=strategy.rp.storage_policy.buffer, strategy=strategy)
        if custom_selection_result:
            self.selection_result = custom_selection_result
        else:
            self.selection_result = al_strategy.query(round(len(strategy.experience.dataset)*self.fraction))
        strategy.experience.dataset = strategy.experience.dataset.subset(np.sort(self.selection_result))
        print(len(strategy.experience.dataset))
        self.model.clf.train()
