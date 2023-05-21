import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
import random


def get_data(dataset, data_path, batch_size, num_workers,train_shuffle=True):
    print("Loading dataset {} from {}".format(dataset, data_path))
    ds = getattr(datasets, dataset.upper())
    path = os.path.join(data_path)
    if dataset == 'SVHN':
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
                ),
            ]
        )
        transform_test = transform_train
        train_set = ds(path, split='train', download=True, transform=transform_train)
        test_set = ds(path, split='test', download=True, transform=transform_test)
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_set = ds(path, train=True, download=True, transform=transform_train)
        test_set = ds(path, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )

    return trainloader,testloader