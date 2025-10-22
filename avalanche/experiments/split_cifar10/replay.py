#!/usr/bin/env python3
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
# from avalanche.benchmarks.generators import benchmark_with_validation_stream
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, confusion_matrix_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models.resnet32 import resnet32
from avalanche.models.slim_resnet18 import SlimResNet18, MTSlimResNet18
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, LRSchedulerPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.supervised import Naive
from experiments.utils import create_default_args, set_seed
from torch.optim.lr_scheduler import StepLR


def replay_scifar10(override_args=None):
    """ 
    Replay for Split CIFAR100
    """

    args = create_default_args(
        {
            "cuda": 0,
            "num_epochs": 20,
            "mem_size": 500,
            "momentum": 0.8,
            "weight_decay": 0.0002,
            "lr": 0.01,
            "train_mb_size": 32,
            "seed": None,
            "batch_size_mem": 32,
        },
        override_args
    )
    set_seed(args.seed)
    fixed_class_order = np.arange(10)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    scenario = SplitCIFAR10(
        5,
        return_task_id=False,
        seed=args.seed,
        fixed_class_order=fixed_class_order,
        shuffle=True,
        class_ids_from_zero_in_each_exp=False,
    )

    # scenario = benchmark_with_validation_stream(scenario, 0.05)
    input_size = (3, 32, 32)
    model = SlimResNet18(nclasses=10)
    model.linear = IncrementalClassifier(model.linear.in_features, 2)

    optimizer = SGD(model.parameters(), momentum=args.momentum, weight_decay=args.weight_decay, lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=args.num_epochs//3, gamma=0.3)

    # scheduler_plugin = LRSchedulerPlugin(scheduler, step_granularity="epoch", first_exp_only=False)

    interactive_logger = InteractiveLogger()

    loggers = [interactive_logger]

    training_metrics = []

    evaluation_metrics = [
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=10, save_image=False, stream=True),
    ]

    evaluator = EvaluationPlugin(
        *training_metrics,
        *evaluation_metrics,
        loggers=loggers,
    )
    
    storage_policy = ClassBalancedBuffer(args.mem_size, adaptive_size=True)
    plugins = [ReplayPlugin(args.mem_size, storage_policy=storage_policy)]

    cl_strategy = Naive(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        evaluator=evaluator,
        device=device,
        train_mb_size=args.train_mb_size,
        train_epochs=args.num_epochs,
        eval_mb_size=64,
    )

    for t, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(
            experience,
            eval_streams=[],
            num_workers=0,
            drop_last=True,
        )

        cl_strategy.eval(scenario.test_stream[: t + 1])

    # Only evaluate at the end on the test stream
    results = cl_strategy.eval(scenario.test_stream)

    return results


if __name__ == "__main__":
    res = replay_scifar10()
    print(res)
