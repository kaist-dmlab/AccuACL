from typing import Optional, Sequence, List, Union
import torch
from torch.nn.parameter import Parameter

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
    BiCPlugin,
    MIRPlugin,
    FromScratchTrainingPlugin,
    ActiveReplayPlugin,
    ActiveMemoryReplayPlugin,
    QueryPlugin,
)
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates import SupervisedTemplate
from avalanche.models.generator import MlpVAE, VAE_loss
from avalanche.logging import InteractiveLogger

class active_GSS(SupervisedTemplate):
    def __init__(self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        al_strategy,
        fraction: float,
        mem_size: int = 200,
        mem_strength=1,
        input_size=[],
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        rp = GSS_greedyPlugin(mem_size=mem_size, mem_strength=mem_strength, input_size=input_size)
        qp = QueryPlugin(al_strategy = al_strategy, fraction = fraction, batch_size = train_mb_size, mem = rp.memory_dataset, device = device)
        if plugins is None:
            plugins = [qp,rp]
        else:
            plugins.extend([qp,rp])
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )



class active_ER(SupervisedTemplate):
    """Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        al_strategy,
        fraction: float,
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        self.rp = ReplayPlugin(mem_size)
        self.qp = QueryPlugin(al_strategy = al_strategy, fraction = fraction, batch_size = train_mb_size, mem = self.rp.storage_policy.buffer,
            device = device
        )
        if plugins is None:
            plugins = [self.qp,self.rp]
        else:
            plugins.extend([self.qp,self.rp])
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )