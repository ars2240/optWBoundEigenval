# forest.py
#
# Author: Adam Sandler
# Date: 1/19/21
#
# Classifies forest cover types
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
from forest_data import get_data, Net


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0.01, 'K': 0}

    # batch size
    batch_size = 128
    opt['batch_size'] = batch_size

    # Load the dataset
    opt.update(get_data())

    def beta(k):
        return 1/(1+k)

    # Training Setup
    opt['model'] = Net()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.SGD(opt['model'].parameters(), lr=.5)
    opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(opt['optimizer'], lr_lambda=beta)
    opt['header'] = 'Cov'
    opt['use_gpu'] = False
    opt['train'] = True
    opt['lobpcg'] = False
    opt['verbose'] = False
    opt['rho_test'] = False
    opt['ignore_bad_vals'] = False

    return opt
