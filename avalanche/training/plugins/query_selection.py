
from typing import Optional, TYPE_CHECKING

import numpy as np
from torch.utils.data.dataloader import DataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class QueryPlugin(SupervisedPlugin):
    """
    query selection plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks.
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    """

    def __init__(
        self,
        al_strategy,
        fraction = 0.2,
        batch_size = 1,
        mem = None,
        device=None,
    ):
        super().__init__()
        self.fraction = fraction
        self.al_strategy = al_strategy
        self.device = device
        self.batch_size = batch_size
        self.mem = mem
        self.selection_result = None

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        # if len(self.storage_policy.buffer) == 0:
        #     # first experience. We don't use the buffer, no need to change
        #     # the dataloader.
        #     return


        assert strategy.experience.dataset is not None
        print(len(strategy.experience.dataset))
        al_strategy = self.al_strategy(strategy.experience.dataset, strategy=strategy, fraction=self.fraction, mem=self.mem)
        self.selection_result = al_strategy.select()
        strategy.experience.dataset = strategy.experience.dataset.subset(np.sort(self.selection_result))
        strategy.adapted_dataset = strategy.experience.dataset
        print(len(strategy.experience.dataset))

        strategy.dataloader = DataLoader(
                strategy.experience.dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
            )