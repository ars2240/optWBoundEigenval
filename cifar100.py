# cifar100.py
#
# Author: Adam Sandler
# Date: 8/11/19
#
# Classifies images from the CIFAR100 dataset
#
#
# Dependencies:
#   Packages: random, numpy, torch, torchvision, scikit-learn
#   Files: opt

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
# import scipy.io as sio
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as tvm
from opt import OptWBoundEignVal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cifar100_data import get_train_valid_loader, get_test_loader

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.005
batch_size = 128
mu = 0
K = 0

# def mu(i):
#    return np.max([0.0, (i-50)/1000])

# Load Data
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# Load the dataset
train_loader, valid_loader, train_loader_na = get_train_valid_loader(batch_size=batch_size, augment=True)
test_loader = get_test_loader(batch_size=batch_size)


# learning rate
def alpha(i):
    if i < 81:
        return 1
    elif i < 122:
        return 0.1
    else:
        return 0.1**2


# Train Neural Network

# Create neural network
model = tvm.resnet50(num_classes=100)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=200,
                       max_pow_iter=10000, verbose=True, header='CIFAR100', use_gpu=True, pow_iter=False)

# Train model
opt.train(loader=train_loader, valid_loader=valid_loader, train_loader=train_loader_na)

opt.test_test_set(loader=test_loader)  # test model on test set

opt.parse()
