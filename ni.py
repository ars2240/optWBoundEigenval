# ni.py
#
# Author: Adam Sandler
# Date: 5/7/19
#
# Classifies network intrusion from the KDD Cup 99 dataset
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
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

# def mu(i):
#    return np.max([0.0, (i-50)/1000])

# Load Data
# u = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
u = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
filename2 = download(u)
u = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
filename3 = download(u)

# import dataset
train = pd.read_csv(filename2, header=None)
test = pd.read_csv(filename3, header=None)

# transform to supercategories
dic = {'normal.': 'normal',  'nmap.': 'probing', 'portsweep.': 'probing', 'ipsweep.': 'probing', 'satan.': 'probing',
       'land.': 'dos', 'pod.': 'dos', 'teardrop.': 'dos', 'back.': 'dos', 'neptune.': 'dos', 'smurf.': 'dos',
       'spy.': 'r2l', 'phf.': 'r2l', 'multihop.': 'r2l', 'ftp_write.': 'r2l', 'imap.': 'r2l', 'warezmaster.': 'r2l',
       'guess_passwd.': 'r2l', 'buffer_overflow.': 'u2r', 'rootkit.': 'u2r', 'loadmodule.': 'u2r', 'perl.': 'u2r'}
i = train.shape[1] - 1
train = train.loc[train[i].isin(dic.keys())]
train.replace({i: dic}, inplace=True)
test = test.loc[test[i].isin(dic.keys())]
test.replace({i: dic}, inplace=True)

train_len = train.shape[0]  # save length of training set
train = train.append(test, ignore_index=True)
inputs = pd.get_dummies(train)  # convert objects to one-hot encoding

X = inputs.values[:train_len, :-5]
y_onehot = inputs.values[:train_len, -5:]
y = np.asarray([np.where(r == 1)[0][0] for r in y_onehot])  # convert from one-hot to integer encoding

# class balance
cts = []
for i in set(y):
    cts.append(list(y).count(i))
print(cts/np.sum(cts))

X_test = inputs.values[train_len:, :-5]
y_test_onehot = inputs.values[train_len:, -5:]
y_test = np.asarray([np.where(r == 1)[0][0] for r in y_test_onehot])  # convert from one-hot to integer encoding

# class balance
cts = []
for i in set(y_test):
    cts.append(list(y_test).count(i))
print(cts/np.sum(cts))

X = X.reshape((train_len, 123))
X_test = X_test.reshape((test.shape[0], 123))

# normalize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=1/5, random_state=1226)

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
        self.fc1 = nn.Linear(123, 13)
        self.fc2 = nn.Linear(13, 15)
        self.fc3 = nn.Linear(15, 20)
        self.fc4 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=0)
        return x


alpha = lambda k: 1/(1+k)

# Train Neural Network

# Create neural network
model = Net()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='NI', use_gpu=True)

# Train model
opt.train(X, y, X_valid, y_valid)

opt.test_test_set(X_test, y_test)  # test model on test set
