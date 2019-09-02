# cifar10.py
#
# Author: Adam Sandler
# Date: 9/30/19
#
# Classifies images from the CIFAR10 dataset
#
#
# Dependencies:
#   Packages: random, numpy, torch, torchvision, scikit-learn
#   Files: opt

import sys
import os
import numpy as np
import torch
import torch.utils.data as utils_data
# import scipy.io as sio
import torchvision.models as tvm
from opt import OptWBoundEignVal
from cifar10_data import get_norm, get_train_valid_loader, get_test_loader

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
train_loader, valid_loader = get_train_valid_loader(batch_size=batch_size, augment=False)
test_loader = get_test_loader(batch_size=batch_size)


# learning rate
def alpha(i):
    if i < 60:
        return 1
    elif i < 120:
        return 0.2
    elif i < 160:
        return 0.2**2
    else:
        return 0.2**3


# Train Neural Network

# Create neural network
model = tvm.resnet50(num_classes=10)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=200,
                       max_pow_iter=10000, verbose=False, header='CIFAR10', use_gpu=True, pow_iter=False)

# Train model
opt.train(loader=train_loader, valid_loader=valid_loader)
opt.test_test_set(loader=test_loader)  # test model on test set
opt.parse()
