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
from sam import SAM


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}

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
    base_optimizer = torch.optim.SGD
    params = [{'params': opt['model'].parameters()}]
    opt['optimizer'] = SAM(params, base_optimizer, lr=0.5)
    opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(opt['optimizer'], lr_lambda=beta)
    opt['header'] = 'Cov'
    opt['use_gpu'] = False
    opt['train'] = True
    opt['lobpcg'] = False
    opt['verbose'] = True
    opt['pow_iter'] = False
    opt['rho_test'] = True
    opt['ignore_bad_vals'] = False

    return opt
