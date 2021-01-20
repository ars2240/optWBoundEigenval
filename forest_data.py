"""
Create train, valid, test iterators for USPS
Based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

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

def get_data(u = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'):
    # Load Data
    filename2 = download(u)

    # import dataset
    data = pd.read_csv(filename2, header=None)

    #print(set(data.values[:, -1]))

    X = data.values[:, :-1]
    y = data.values[:, -1] - 1

    # class balance
    #cts = []
    #for i in set(y):
    #    cts.append(list(y).count(i))
    #print(cts)

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

    return {'inputs': X, 'target': y, 'inputs_valid': X_valid, 'target_valid': y_valid, 'inputs_test': X_test,
            'target_test': y_test}

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