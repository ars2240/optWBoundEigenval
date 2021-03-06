# usps.py
#
# Author: Adam Sandler
# Date: 1/19/21
#
# Classifies digits from the USPS dataset
#
#
# Dependencies:
#   Packages: random, numpy, torch, requests, gzip, shutil, pandas
#   Files: opt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scipy.io as sio
from usps_data import get_train_valid_loader, get_test_loader
from optim import EntropySGD


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}

    # batch size
    batch_size = 128
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    opt['train_loader'], opt['valid_loader'] = get_train_valid_loader(batch_size=batch_size, augment=False)
    opt['test_loader'] = get_test_loader(batch_size=batch_size)
    opt['test_loader_aug'] = get_test_loader(batch_size=batch_size, augment=True)

    # Feed-Forward NN
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            n = 20
            self.fc1 = nn.Linear(256, n)
            self.fc2 = nn.Linear(n, n)
            self.fc3 = nn.Linear(n, 10)
            self.bn1 = nn.BatchNorm1d(n)

        def forward(self, x):
            x = self.bn1(F.relu(self.fc1(x)))
            x = self.bn1(F.relu(self.fc2(x)))
            x = self.bn1(F.relu(self.fc2(x)))
            x = self.fc3(x)
            x = F.softmax(x, dim=1)
            return x

    # CNN model
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

    # learning rate
    def alpha(k):
        return 1 / (1 + np.sqrt(k))

    # Training Setup
    opt['model'] = CNN()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = EntropySGD(opt['model'].parameters())
    # opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(options['optimizer'], lr_lambda=alpha)
    opt['header'] = 'USPS'
    opt['train'] = True
    opt['pow_iter'] = False

    opt['aug_test'] = True

    return opt
