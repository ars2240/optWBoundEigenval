# usps.py
#
# Author: Adam Sandler
# Date: 8/5/19
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
from opt import OptWBoundEignVal, download
import pandas as pd
from sklearn.model_selection import train_test_split

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.001
batch_size = 128
mu = 0.005
K = 0


# def mu(i):
#    return np.max([0.0, (i-50)/1000])


# Load Data
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# Load Data
u = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz'
filename2 = download(u)
u = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz'
filename3 = download(u)

# import dataset
train = pd.read_csv(filename2, header=None, sep=' ')
test = pd.read_csv(filename3, header=None, sep=' ')

X = np.asarray(-train.loc[:, 1:256])
y = np.asarray(train.loc[:, 0])
X_test = np.asarray(-test.loc[:, 1:256])
y_test = np.asarray(test.loc[:, 0])

X = X.reshape((7291, 256))
X_test = X_test.reshape((2007, 256))

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=1/7, random_state=1226)

# normalization
X_m = X.mean()
X_sd = X.std()
X = (X-X_m)/X_sd
X_valid = (X_valid-X_m)/X_sd
X_test = (X_test-X_m)/X_sd

# convert data-types
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()
X_valid = torch.from_numpy(X_valid).float()
y_valid = torch.from_numpy(y_valid).long()


#   Define Neural Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 100
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


alpha = lambda k: 1/(1+k)

# Train Neural Network

# Create neural network
model = Net()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='USPS')

# Train model
opt.train(X, y, X_valid, y_valid)

opt.test_test_set(X_test, y_test)  # test model on test set
