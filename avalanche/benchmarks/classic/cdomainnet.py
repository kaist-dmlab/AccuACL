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

transform = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def DomainNet(data_path, num_tasks, train_ratio=0.8):
    train_experiences=[]
    test_experiences=[]
    for task, path in enumerate([name for name in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, name))]):
        print(path)
        classes_list = sorted(os.listdir(os.path.join(data_path, path)))[:100]
        for label, class_path in enumerate([os.path.join(data_path, path, name) for name in classes_list if os.path.isdir(os.path.join(data_path, path, name))]):
            filenames_list = os.listdir(class_path)
            experience_paths = []
            # print(len(filenames_list))
            for name in filenames_list:
                instance_tuple = (os.path.join(data_path, class_path, name), label)
                experience_paths.append(instance_tuple)
            random.shuffle(experience_paths)
            train_len = int(len(experience_paths)*train_ratio)
            train_experiences.append(experience_paths[:train_len])
            test_experiences.append(experience_paths[train_len:])
    sublist_size = len(train_experiences) // num_tasks
    train_experiences = [flatten_list(train_experiences[i * sublist_size : (i + 1) * sublist_size]) for i in range(num_tasks)]
    test_experiences = [flatten_list(test_experiences[i * sublist_size : (i + 1) * sublist_size]) for i in range(num_tasks)]

    return paths_benchmark(
        train_experiences,
        test_experiences,
        task_labels = [0]*num_tasks,
        train_transform=transform,
        eval_transform=transform,
        # other_streams_transforms={'strong': strong_transform}
    )

if __name__ == "__main__":
    import sys

    benchmark_instance = DomainNet(
        '/data3/jhpark/DomainNet',
        6
    )
    check_vision_benchmark(benchmark_instance, show_without_transforms=False)
    sys.exit(0)
