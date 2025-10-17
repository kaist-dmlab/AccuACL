################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" This module contains the high-level CORe50 benchmark generator. It
basically returns a iterable benchmark object ``GenericCLScenario`` given a
number of configuration parameters."""
from pathlib import Path
from typing import Union, Optional, Any

from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from avalanche.benchmarks import nc_benchmark, NCScenario

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.scenarios.generic_benchmark_creation import (
    create_generic_benchmark_from_filelists,
)
from avalanche.benchmarks.datasets.core50.core50 import CORe50Dataset

nbatch = {
    "ni": 8,
    "nc": 9,
    "nic": 79,
    "nicv2_79": 79,
    "nicv2_196": 196,
    "nicv2_391": 391,
}

scen2dirs = {
    "ni": "batches_filelists/NI_inc/",
    "nc": "batches_filelists/NC_inc/",
    "nic": "batches_filelists/NIC_inc/",
    "nicv2_79": "NIC_v2_79/",
    "nicv2_196": "NIC_v2_196/",
    "nicv2_391": "NIC_v2_391/",
}


normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_default_train_transform = Compose(
    [ToTensor(), RandomHorizontalFlip(), normalize]
)

_default_eval_transform = Compose([ToTensor(), normalize])


def SplitCORe50(
    n_experiences: int,
    *,
    object_lvl: bool = True,
    mini: bool = False,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    input_size = None,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root: Optional[Union[str, Path]] = None
):
    """
    Creates a CL benchmark for CORe50.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    This generator can be used to obtain the NI, NC, NIC and NICv2-* scenarios.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The task label 0 will be assigned to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param scenario: CORe50 main scenario. It can be chosen between 'ni', 'nc',
        'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391.'
    :param run: number of run for the benchmark. Each run defines a different
        ordering. Must be a number between 0 and 9.
    :param object_lvl: True for a 50-way classification at the object level.
        False if you want to use the categories as classes. Default to True.
    :param mini: True for processing reduced 32x32 images instead of the
        original 128x128. Default to False.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dataset_root: Absolute path indicating where to store the dataset
        and related metadata. Defaults to None, which means that the default
        location for
        'core50' will be used.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location("core50")

    # Download the dataset and initialize filelists
    # core_data = CORe50Dataset(root=dataset_root, mini=mini, object_level=object_lvl)
    train_data = CORe50Dataset(root=dataset_root, mini=mini, object_level=object_lvl)
    test_data = CORe50Dataset(train=False, root=dataset_root, mini=mini, object_level=object_lvl)

    scenario =nc_benchmark(
        train_dataset=train_data,
        test_dataset=test_data,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    return scenario


__all__ = ["SplitCORe50"]

if __name__ == "__main__":
    import sys

    benchmark_instance = SplitCORe50(10)
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)
