# cov.py
#
# Author: Adam Sandler
# Date: 5/7/19
#
# Classifies forest cover type from UCI data
# https://archive.ics.uci.edu/ml/datasets/covertype
#
#
# Dependencies:
#   Packages: random, numpy, torch, torchvision, requests, gzip, shutil, pandas
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.001
batch_size = 128
mu = 0
K = 0

"""
def mu(ep):
    return np.max([0.0, (ep-49)/50000])
"""

# Load Data
u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
filename2 = download(u)

# import dataset
data = pd.read_csv(filename2, header=None)

print(set(data.values[:, -1]))

X = data.values[:, :-1]
y = data.values[:, -1] - 1

# class balance
cts = []
for i in set(y):
    cts.append(list(y).count(i))
print(cts)

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=1/5, random_state=1226)

X = X.reshape((X.shape[0], 54))
X_test = X_test.reshape((X_test.shape[0], 54))

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=1/5, random_state=1226)

# normalize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# convert data-types
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()
X_valid = torch.from_numpy(X_valid).float()
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


# Neural Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n = 20
        self.fc1 = nn.Linear(54, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


alpha = lambda k: 1/(1+k)

# Train Neural Network

# Create neural network
model = Net()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='Cov')

# Train model
#opt.train(X, y, X_valid, y_valid)

#opt.test_test_set(X_test, y_test)  # test model on test set


if not os.path.exists('./logs'):
    os.mkdir('./logs')

feat = 54
old_stdout = sys.stdout  # save old output
log_file = open("./logs/" + opt.header2 + "_cov_shift.log", "w")  # open log file
sys.stdout = log_file  # write to log file
for i in range(0, feat):
    test_mean = [0.0] * feat
    test_mean[i] = 0.1
    print('Feature %d' % (i+1))
    opt.test_cov_shift(X_test, y_test, test_mean=test_mean)  # test model on test set

log_file.close()  # close log file
sys.stdout = old_stdout  # reset output
