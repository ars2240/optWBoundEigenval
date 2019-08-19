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
mu = 0.1
K = 2


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
print('Mean: ' + str(X_m) + ', StDev: ' + str(X_sd))

# convert data-types
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()
X_valid = torch.from_numpy(X_valid).float()
y_valid = torch.from_numpy(y_valid).long()
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


alpha = lambda k: 1/(1+np.sqrt(k))

# Train Neural Network

# Create neural network
model = CNN()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=200,
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

