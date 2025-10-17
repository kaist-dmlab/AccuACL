import os
import random
from pathlib import Path
from typing import Sequence, Optional, Union, Any

from torchvision import transforms
from acl.augment import RandAugment

from torchvision.transforms import Compose, ToTensor, Resize

from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)


from avalanche.benchmarks import nc_benchmark, NCScenario, paths_benchmark

def build_transform(is_train, input_size=64,):
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ))
    
    return transforms.Compose(t)

def strong_transform(input_size=224):
    resize_im = input_size>32
    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(RandAugment(3,5))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(t)


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def ImageNet_R(data_path, num_tasks, train_ratio=0.8):
    train_experiences=[]
    test_experiences=[]
    for label, path in enumerate([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]):
        filenames_list = os.listdir(os.path.join(data_path, path))
        experience_paths = []
        for name in filenames_list:
            instance_tuple = (os.path.join(data_path, path, name), label)
            experience_paths.append(instance_tuple)
        random.shuffle(experience_paths)
        train_len = int(len(experience_paths)*train_ratio)
        train_experiences.append(experience_paths[:train_len])
        test_experiences.append(experience_paths[train_len:])
    num_tasks = 10
    sublist_size = len(train_experiences) // num_tasks
    train_experiences = [flatten_list(train_experiences[i * sublist_size : (i + 1) * sublist_size]) for i in range(num_tasks)]
    test_experiences = [flatten_list(test_experiences[i * sublist_size : (i + 1) * sublist_size]) for i in range(num_tasks)]

    return paths_benchmark(
        train_experiences,
        test_experiences,
        task_labels = [0]*num_tasks,
        train_transform=build_transform(True),
        eval_transform=build_transform(False),
        other_streams_transforms={'strong': strong_transform}
    )

if __name__ == "__main__":
    import sys

    benchmark_instance = ImageNet_R(
        '/data/jhpark/imagenet-r',
        10
    )
    check_vision_benchmark(benchmark_instance, show_without_transforms=False)
    sys.exit(0)
