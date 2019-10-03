# chestxray.py
#
# Author: Adam Sandler
# Date: 10/2/19
#
# Classifies images from the chest x-ray dataset
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
from opt import OptWBoundEignVal

sys.path.insert(1, '/home/hddraid/shared_data/chest_xray8/code/VClassifier/')

from dcnn import *

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

# Load the dataset
train_set = ChestXray_Dataset(use='train', transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
validation_set = ChestXray_Dataset(use='validation', transform=transform)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
test_set = ChestXray_Dataset(use='test', transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)


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
enc = 'res50'

if enc == 'alex':
    model = MyAlexNet(14)
elif enc == 'res50':
    model = MyResNet50(14)
elif enc == 'vgg16bn':
    model = MyVggNet16_bn(14)
elif enc == 'dens161':
    model = MyDensNet161(14)
elif enc == 'dens201':
    model = MyDensNet201(14)
elif enc == 'dens121':
    model = MyDensNet121(14)

loss = W_BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha)

opt = OptWBoundEignVal(model, loss, optimizer, scheduler, batch_size=batch_size, eps=-1, mu=mu, K=K, max_iter=200,
                       max_pow_iter=10000, verbose=False, header='chestxray_res50', use_gpu=True, pow_iter=False)

# Train model
opt.train(loader=train_loader, valid_loader=valid_loader, train_loader=train_loader_na)
opt.test_test_set(loader=test_loader)  # test model on test set
opt.parse()
