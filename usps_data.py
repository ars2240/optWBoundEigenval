"""
Create train, valid, test iterators for USPS
Based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

normalize = transforms.Normalize(
    mean=[0.24510024090766908],
    std=[0.29806136205673217],
)

trans = transforms.Compose([
            transforms.ToTensor(),
            normalize])

aug_trans = [transforms.Compose([
            transforms.RandomCrop(16, padding=1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize]),transforms.Compose([
            transforms.RandomCrop(16, padding=2),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize])]

mnist_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((16, 16)),
    normalize])

gan_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    normalize])


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


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

    torch.manual_seed(random_seed)

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


def get_test_loader(data_dir='./data', batch_size=1, augment=False, random_seed=1226, shuffle=False, num_workers=0,
                    pin_memory=True):
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

    torch.manual_seed(random_seed)

    if type(transform) is list:
        data_loader = []
        for t in transform:
            dataset = datasets.USPS(
                root=data_dir, train=False,
                download=True, transform=t)

            data_loader.append(torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory))
    else:
        dataset = datasets.USPS(
            root=data_dir, train=False,
            download=True, transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

    return data_loader


def get_mnist_loader(data_dir='./data', batch_size=1, random_seed=1226, shuffle=False, num_workers=0, pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.
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
    transform = mnist_trans

    torch.manual_seed(random_seed)

    if type(transform) is list:
        data_loader = []
        for t in transform:
            dataset1 = datasets.MNIST(
                root=data_dir, train=True,
                download=True, transform=t)

            dataset2 = datasets.MNIST(
                root=data_dir, train=False,
                download=True, transform=t)

            # combine train & test datasets
            dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

            data_loader.append(torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory))
    else:
        dataset1 = datasets.MNIST(
            root=data_dir, train=True,
            download=True, transform=transform)

        dataset2 = datasets.MNIST(
            root=data_dir, train=False,
            download=True, transform=transform)

        # combine train & test datasets
        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory)

    return data_loader


def get_gan_loader(data_dir='./data', file='cgan_usps.pt', batch_size=1, random_seed=1226, shuffle=False, num_workers=0,
                   pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the GAN-generated dataset.
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

    torch.manual_seed(random_seed)

    dataset = torch.load(data_dir + '/' + file)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)

    return data_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # convolutional & pooling layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # linear layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        if type(x).__module__ == 'numpy':
            x = torch.from_numpy(x)

        # change shape
        x = x.view(-1, 1, 16, 16)

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # change shape
        x = x.view(-1, 128)

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x