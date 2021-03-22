from enum import Enum

from src.dataset.CIFAR10 import CIFAR10


class SupportedDataset(Enum):
    CIFAR10_Enum = dict(
        dataloader=CIFAR10,
        image_size=(32, 32),  # Used for model FC layer.
        channels=3,
        training_size=50000,
        labels_count=10
    )


MAP_DATASET_TO_ENUM = dict(
    CIFAR10=SupportedDataset.CIFAR10_Enum,
)


def get_dataset_class(dataset_name):
    if dataset_name not in MAP_DATASET_TO_ENUM:
        raise ValueError('Unsupported Dataset')
    return MAP_DATASET_TO_ENUM[dataset_name].value['dataloader']


def create_dataset(dataset_args, train_data_args, val_data_args):
    dataset_loader_class = get_dataset_class(dataset_args['dataset'])
    return dataset_loader_class(dataset_args, train_data_args, val_data_args)
