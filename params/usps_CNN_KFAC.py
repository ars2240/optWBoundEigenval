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
from usps_data import *
from kfac import KFACOptimizer


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
    opt['test_loader'] = []
    # opt['test_loader'].append(get_test_loader(batch_size=batch_size))
    # opt['test_loader'].append(get_mnist_loader(batch_size=batch_size))
    opt['test_loader'].append(get_gan_loader(batch_size=batch_size, file='constructed6.pt'))
    opt['test_loader_aug'] = get_test_loader(batch_size=batch_size, augment=True)

    # Training Setup
    opt['model'] = CNN()
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = KFACOptimizer(opt['model'], lr=1e-3)
    opt['header'] = 'USPS_E-3'
    opt['use_gpu'] = False
    opt['train'] = True
    opt['btch_h'] = False

    opt['test'] = False
    opt['comp_test'] = False
    opt['aug_test'] = False
    opt['rho_test'] = False
    opt['ignore_bad_vals'] = False
    opt['verbose'] = True

    return opt
