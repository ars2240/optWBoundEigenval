# cifar10 parameter file
#
# Author: Adam Sandler
# Date: 4/14/20
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
    opt = {'seed': 1226, 'tol': 0.01, 'mu': 0.001, 'K': 0}

    # batch size
    batch_size = 32
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    opt['train_loader'], opt['valid_loader'], opt['train_loader_na'] = get_train_valid_loader(batch_size=batch_size,
                                                                                              augment=True)
    opt['test_loader'] = get_test_loader(batch_size=batch_size)

    # learning rate
    def alpha(i):
        if i < 60:
            return 1
        elif i < 80:
            return 0.2
        else:
            return 0.2 ** 2

    # Training Setup
    opt['model'] = DenseNet3(depth=40, growth_rate=12, num_classes=10)
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.SGD(opt['model'].parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(opt['optimizer'], lr_lambda=alpha)
    opt['header'] = 'CIFAR10_DenseNet'
    opt['use_gpu'] = True
    opt['verbose'] = False
    opt['pow_iter_eps'] = 5e-2
    opt['max_pow_iter'] = 100

    return opt
