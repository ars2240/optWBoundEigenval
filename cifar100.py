# cifar100.py
#
# Author: Adam Sandler
# Date: 5/7/19
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

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.001
batch_size = 16
mu = 0
K = 0

# def mu(i):
#    return np.max([0.0, (i-50)/1000])

# Load Data
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# Load the dataset
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# if not exist, download cifar100 dataset
train_set = utils_data.DataLoader(dset.CIFAR100(root=root, train=True, transform=trans, download=True), batch_size=50000)
test_set = utils_data.DataLoader(dset.CIFAR100(root=root, train=False, transform=trans, download=True), batch_size=10000)

_, (X, y) = next(enumerate(train_set))
_, (X_test, y_test) = next(enumerate(test_set))

X = X.reshape((50000, 3, 32, 32))
X_test = X_test.reshape((10000, 3, 32, 32))

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=1/5, random_state=1226)

X = torch.from_numpy(X)
y = torch.from_numpy(y).long()
X_valid = torch.from_numpy(X_valid)
y_valid = torch.from_numpy(y_valid).long()

"""
mdict = sio.loadmat('data')
X = mdict['X'].astype('float32').reshape((60000, 784))/255
y = mdict['y'].squeeze()

# split data set into 70% train, 30% test
X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=1226)

# re-format data
X = torch.from_numpy(X)
y = torch.from_numpy(y).long()
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).long()
"""""

# train_data = utils_data.TensorDataset(X, y)

#   Define Neural Network Architecture
#
#   Modify your neural network here!


alpha = lambda k: 1/(1+np.sqrt(k))

# Train Neural Network

# Create neural network
model = tvm.resnet50()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='CIFAR100', use_gpu=True, mem_track=True)

# Train model
opt.train(X, y, X_valid, y_valid)

opt.test_test_set(X_test, y_test)  # test model on test set
