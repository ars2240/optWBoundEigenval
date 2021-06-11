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
# import scipy.io as sio
from usps_data import get_train_valid_loader, get_test_loader, CNN


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': .001, 'K': 50, 'Kmin': 50}

    # batch size
    batch_size = 128
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    opt['train_loader'], opt['valid_loader'] = get_train_valid_loader(batch_size=batch_size, augment=False)
    opt['test_loader'] = get_test_loader(batch_size=batch_size)
    opt['test_loader_aug'] = get_test_loader(batch_size=batch_size, augment=True)

    # learning rate
    def alpha(k):
        return np.exp(-4*k)

    # Training Setup
    opt['model'] = CNN()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.Adam(opt['model'].parameters())
    # opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(options['optimizer'], lr_lambda=alpha)
    opt['header'] = 'USPS_LOBPCG4_8'
    opt['use_gpu'] = True
    opt['train'] = True
    opt['lobpcg'] = True
    opt['verbose'] = True
    opt['pow_iter_alpha'] = alpha
    opt['mem_track'] = False
    opt['ignore_bad_vals'] = True

    opt['aug_test'] = True
    opt['rho_test'] = True
    opt['kfac_rand'] = False
    opt['kfac_batch'] = 8

    return opt
