"""
Create train, valid, test iterators for CIFAR-10
Based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

normalize = transforms.Normalize(
    mean=[0.49088515, 0.48185424, 0.44636887],
    std=[0.20222517, 0.19923602, 0.20073999],
)


def get_norm(data_dir='./data', batch_size=1, random_seed=1226, valid_size=0.2, shuffle=False, num_workers=0,
             pin_memory=False):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    b_m = []
    b_sd = []
    b_size = []

    for _, data in enumerate(train_loader):
        inputs, target = data

        b_m.extend(inputs.mean(dim=[0, 2, 3]))
        b_sd.extend(inputs.std(dim=[0, 2, 3]))

        b_size.append(len(target))

    b_m = np.reshape(b_m, [len(b_size), 3])
    b_sd = np.reshape(b_sd, [len(b_size), 3])
    m = np.average(b_m, weights=b_size, axis=0)
    sd = np.average(b_sd, weights=b_size, axis=0)

    return m, sd


def get_train_valid_loader(data_set = '100', data_dir='./data', batch_size=1, augment=False, random_seed=1226, valid_size=0.2,
                           shuffle=False, show_sample=False, num_workers=1, pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10/100 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(0, translate=(1/32, 1/32)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    if data_set.endswith('100'):
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True,
                                          transform=train_transform)
        valid_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True,
                                          transform=valid_transform)
    elif data_set.endswith('10'):
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                         transform=train_transform)
        valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                         transform=valid_transform)
    else:
        raise Exception('Invalid dataset selected.')

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        from utils import plot_images
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    if augment:
        # non-augmented training dataset
        if data_set.endswith('100'):
            train_dataset_na = datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                 transform=valid_transform)
        elif data_set.endswith('10'):
            train_dataset_na = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                transform=valid_transform)
        train_loader_na = torch.utils.data.DataLoader(
            train_dataset_na, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, valid_loader, train_loader_na
    else:
        return train_loader, valid_loader


def get_test_loader(data_set='100', data_dir='./data', batch_size=1, shuffle=False, num_workers=1, pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10/100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if data_set.endswith('100'):
        dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    elif data_set.endswith('10'):
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    else:
        raise Exception('Invalid dataset selected.')

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

