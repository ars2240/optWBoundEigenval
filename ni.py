"""
Classifies network intrusion from the KDD Cup 99 dataset
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils_data
# import scipy.io as sio
from opt import OptWBoundEignVal
from sklearn.model_selection import train_test_split
import requests
import gzip
import shutil
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
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

# Download and parse the dataset
url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
filename = root + '/' + url.split("/")[-1]
exists = os.path.isfile(filename)
if not exists:
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
filename2 = filename[:-3] + '.csv'
exists = os.path.isfile(filename2)
if not exists:
    with gzip.open(filename, 'rb') as f_in:
        with open(filename2, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
url = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
filename = root + '/' + url.split("/")[-1]
exists = os.path.isfile(filename)
if not exists:
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)
filename3 = filename[:-3] + '.csv'
exists = os.path.isfile(filename3)
if not exists:
    with gzip.open(filename, 'rb') as f_in:
        with open(filename3, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# import dataset
train = pd.read_csv(filename2, header=None)
test = pd.read_csv(filename3, header=None)
for i in range(0, train.shape[1]):
    if train.dtypes[i] == 'object':
        s = list(set(train.values[:, i]))
        dic = {s[i]: i for i in range(0, len(s))}
        train.replace({i: dic}, inplace=True)
        test.replace({i: dic}, inplace=True)

X = train.values[:, :-1]
y = train.values[:, -1]

X_test = test.values[:, :-1]
y_test = test.values[:, -1]

X = X.reshape((494021, 41))
X_test = X_test.reshape((311029, 41))

X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=1/5, random_state=1226)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
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
        self.fc1 = nn.Linear(41, 13)
        self.fc2 = nn.Linear(13, 15)
        self.fc3 = nn.Linear(15, 20)
        self.fc4 = nn.Linear(20, 23)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=0)
        return x


alpha = lambda k: 1/(1+np.sqrt(k))

# Train Neural Network

# Create neural network
model = Net()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                       max_pow_iter=10000, verbose=False, header='NI', use_gpu=True)

# Train model
opt.train(X, y, X_valid, y_valid)

opt.test_test_set(X_test, y_test)  # test model on test set
