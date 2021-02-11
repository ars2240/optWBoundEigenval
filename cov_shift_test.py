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
from opt import OptWBoundEignVal, download, cov_shift_tester
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from forest_data import get_data, Net
from optim import EntropySGD
from kfac import KFACOptimizer
from asymmetric_valley import AsymmetricValley

# set seed
np.random.seed(1226)
torch.manual_seed(1226)

# Parameters
tol = 0.001
batch_size = 128

dic = get_data()

alpha = lambda k: 1/(1+k)

header = 'Cov_norm_5'
indices = "./logs/" + header + "_cov_shift_indices.csv"

# Create neural network
model = Net()
loss = torch.nn.CrossEntropyLoss()

"""
optimizer = torch.optim.SGD(model.parameters(), lr=.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

mu = 0.01
K = 1
opt1 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

mu = 0.01
K = 0
opt2 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

mu = 0.001
K = 5
opt3 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

mu = 0.001
K = 0
opt4 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

mu = 0.005
K = 1
opt5 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

mu = 0
K = 0
opt6 = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Cov')

optimizer = EntropySGD(model.parameters())
mu = 0
K = 0
opt7 = OptWBoundEignVal(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Forest')


optimizer = KFACOptimizer(model)
mu = 0
K = 0
opt8 = OptWBoundEignVal(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Forest')
"""

optimizer = torch.optim.SGD(model.parameters(), lr=.5)
mu = 0
K = 0
opt9 = AsymmetricValley(model, loss, optimizer, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=100,
                        max_pow_iter=10000, verbose=False, header='Forest')

models = [opt9]
#models = [opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9]

cov_shift_tester(models, x=dic['inputs_test'], y=dic['target_test'], iters=1000, bad_modes=[28],
                 header=header, mult=.05, mean_diff=1, indices=indices, append=True)
