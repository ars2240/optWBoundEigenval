"""
Create train, valid, test iterators for USPS
Based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

normalize = transforms.Normalize(
    mean=[0.24510024090766908],
    std=[0.29806136205673217],
)

trans = transforms.Compose([
            transforms.ToTensor(),
            normalize])

aug_trans = transforms.Compose([
            transforms.RandomCrop(16, padding=2),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize])


def get_train_valid_loader(data_dir='./data', batch_size=1, augment=False, random_seed=1226, valid_size=1/7,
                           shuffle=False, show_sample=False, num_workers=0, pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the USPS dataset. A sample
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
    valid_transform = trans
    if augment:
        train_transform = aug_trans
    else:
        train_transform = trans

    # load the dataset
    train_dataset = datasets.USPS(
        root=data_dir, train=True,
        download=True, transform=train_transform)

    valid_dataset = datasets.USPS(
        root=data_dir, train=True,
        download=True, transform=valid_transform)

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
        train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    # visualize some images
    if show_sample:
        from utils import plot_images
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    if augment:
        # non-augmented training dataset
        train_dataset_na = datasets.USPS(
            root=data_dir, train=True,
            download=True, transform=valid_transform)
        train_loader_na = torch.utils.data.DataLoader(
            train_dataset_na, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, valid_loader, train_loader_na
    else:
        return train_loader, valid_loader


def get_test_loader(data_dir='./data', batch_size=1, augment=False, shuffle=False, num_workers=1, pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the USPS dataset.
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
    if augment:
        transform = aug_trans
    else:
        transform = trans

    dataset = datasets.USPS(
        root=data_dir, train=False,
        download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)

    return data_loader

