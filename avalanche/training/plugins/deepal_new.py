
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



class CAL_Plugin(SupervisedPlugin):

    def __init__(
        self,
    ):
        super().__init__()

    def after_train_dataset_adaptation(
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

        print(len(strategy.adapted_dataset))
        if custom_selection_result is not None:
            strategy.adapted_dataset = strategy.adapted_dataset.subset(np.sort(custom_selection_result))
        print(len(strategy.adapted_dataset))

