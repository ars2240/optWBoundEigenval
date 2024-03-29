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
from usps_data import *


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': .005, 'K': 0}

    # batch size
    batch_size = 128
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # Load the dataset
    opt['train_loader'], opt['valid_loader'] = get_train_valid_loader(batch_size=batch_size, augment=False)
    opt['test_loader'] = []
    opt['test_loader'].append(get_test_loader(batch_size=batch_size))
    opt['test_loader'].append(get_mnist_loader(batch_size=batch_size))
    opt['test_loader'].append(get_gan_loader(batch_size=batch_size, file='constructed2.pt'))
    opt['test_loader'].append(get_gan_loader(batch_size=batch_size, file='constructed6.pt'))
    opt['test_loader_aug'] = get_test_loader(batch_size=batch_size, augment=True)

    # learning rate
    def alpha(k):
        return np.exp(-4*k)

    # Training Setup
    opt['model'] = CNN()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.Adam(opt['model'].parameters(), lr=1e-4)
    # opt['scheduler'] = torch.optim.lr_scheduler.LambdaLR(options['optimizer'], lr_lambda=alpha)
    opt['header'] = 'USPS_LOBPCG4'
    opt['max_iter'] = 100
    opt['use_gpu'] = False
    opt['lobpcg'] = True
    opt['verbose'] = False
    opt['pow_iter_alpha'] = alpha
    opt['mem_track'] = False
    opt['ignore_bad_vals'] = True
    opt['train'] = True
    opt['btch_h'] = False

    opt['test'] = False
    opt['comp_test'] = False
    opt['aug_test'] = False
    opt['rho_test'] = False
    opt['ignore_bad_vals'] = False

    opt['kfac_rand'] = False
    opt['kfac_batch'] = 4

    return opt
