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
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}

    # batch size
    batch_size = 128
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    opt.update(get_data())

    # Training Setup
    opt['model'] = Net()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.SGD(opt['model'].parameters(), lr=0.5)
    # opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(options['optimizer'], lr_lambda=alpha)
    opt['header'] = 'Forest_AsymValley'
    opt['train'] = True
    opt['pow_iter'] = False
    opt['asymmetric_valley'] = True
    opt['rho_test'] = False
    opt['ignore_bad_vals'] = False

    return opt
