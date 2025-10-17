import dill

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.food101 import Food101

def get_food101_dataset(dataset_root, input_size):
    if dataset_root is None:
        dataset_root = default_dataset_location("food101")

    train_set = Food101(str(dataset_root), img_size=input_size, split='train', download=True)
    test_set = Food101(str(dataset_root), img_size=input_size, split='test', download=True)

    return train_set, test_set

__all__ = [
    'get_food101_dataset'
]