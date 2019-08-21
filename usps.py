# usps.py
#
# Author: Adam Sandler
# Date: 8/19/19
#
# Classifies digits from the USPS dataset
#
#
# Dependencies:
#   Packages: random, numpy, torch, requests, gzip, shutil, pandas
#   Files: opt

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
# import scipy.io as sio
from opt import OptWBoundEignVal
from usps_data import get_train_valid_loader, get_test_loader

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.001
batch_size = 128
mu = 0.04
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

"""
size = []
m = []
sd = []
for _, data in enumerate(train_loader):
    inputs, outputs = data

    size.append(len(outputs))
    m.append(inputs.mean())
    sd.append(inputs.std())

print(np.average(m, weights=size))
print(np.average(sd, weights=size))
"""


#   Define Neural Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 20
        self.fc1 = nn.Linear(256, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 10)
        self.bn1 = nn.BatchNorm1d(n)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn1(F.relu(self.fc2(x)))
        x = self.bn1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # convolutional & pooling layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # linear layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # change shape
        x = x.view(-1, 1, 16, 16)

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # change shape
        x = x.view(-1, 128)

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def alpha(k):
    return 1/(1+np.sqrt(k))


# Training Setup
model = CNN()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='USPS', use_gpu=False)

# Train model
opt.train(loader=train_loader, valid_loader=valid_loader)
opt.test_test_set(loader=test_loader)  # test model on test set
opt.parse()

# Augmented Testing
test_loader = get_test_loader(batch_size=batch_size, augment=True)
_, acc, f1 = opt.test_model_best(loader=test_loader)
print('Aug_Test_Acc\tAug_Test_F1')
print(str(acc) + '\t' + str(f1))

