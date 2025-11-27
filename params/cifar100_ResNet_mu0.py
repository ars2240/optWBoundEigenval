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

import os
import numpy as np
import sys
import torch
#sys.path.insert(0, '/home/hddraid/shared_data/chest_xray8/code/VClassifier/')  # add folder containing dcnn to path
from dcnn import *
from cifar_data import *


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}

    # batch size
    batch_size = 32
    opt['batch_size'] = batch_size

    # Load the dataset
    opt['train_loader'], opt['valid_loader'], opt['train_loader_na'] = get_train_valid_loader(batch_size=batch_size)
    opt['test_loader'] = get_test_loader(batch_size=batch_size)

    # Training Setup
    opt['model'] = MyResNet50(100)
    opt['loss'] = nn.CrossEntropyLoss()
    opt['optimizer'] = torch.optim.Adam(opt['model'].parameters())
    opt['header'] = 'CIFAR100_ResNet'
    opt['pow_iter'] = False
    opt['train'] = True
    opt['btch_h'] = False

    opt['test'] = True
    opt['rho_test'] = True
    opt['ignore_bad_vals'] = False

    return opt
