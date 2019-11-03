# usps.py
#
# Author: Adam Sandler
# Date: 11/1/19
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
from cifar10_data import get_norm, get_train_valid_loader, get_test_loader
from densenet import DenseNet3


def options():
    # create options dictionary and some parameters
    options = {'seed': 1226, 'tol': 0.001, 'mu': 0.001, 'K': 0}

    # batch size
    batch_size = 64
    options['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    options['train_loader'], options['valid_loader'], options['train_loader_na'] = get_train_valid_loader(
        batch_size=batch_size, augment=True)
    options['test_loader'] = get_test_loader(batch_size=batch_size)

    # learning rate
    def alpha(i):
        if i < 150:
            return 1
        elif i < 225:
            return 0.1
        else:
            return 0.1 ** 2

    # Training Setup
    options['model'] = DenseNet3(depth=40, growth_rate=12, num_classes=10)
    options['loss'] = nn.CrossEntropyLoss()
    options['optimizer'] = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    options['scheduler'] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)
    options['header'] = 'CIFAR10_DenseNet'
    options['use_gpu'] = True

    return options
