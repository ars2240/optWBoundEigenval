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
from opt import OptWBoundEignVal
sys.path.insert(0, '/home/hddraid/shared_data/chest_xray8/code/VClassifier/')  # add folder containing dcnn to path
from dcnn import *


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}
    enc = 'alex'  # model type

    # batch size
    batch_size = 2
    opt['batch_size'] = batch_size

    # def mu(i):
    #    return np.max([0.0, (i-50)/1000])

    # normalize images
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load the dataset
    train_set = ChestXray_Dataset(use='train', transform=transform)
    opt['train_loader'] = DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    valid_set = ChestXray_Dataset(use='validation', transform=transform)
    opt['valid_loader'] = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    test_set = ChestXray_Dataset(use='test', transform=transform)
    opt['test_loader'] = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

    t = torch.zeros((1, 14)).to('cuda')
    n = 0
    for _, data in enumerate(train_loader):
        target = Variable(data['label'].to('cuda'))
        t = torch.sum(torch.cat((target, t)), dim=0).unsqueeze(0)
        n += len(target)
    print(t.to('cpu'))
    print(n)

    # Create neural network
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

    # Training Setup
    opt['model'] = DenseNet3(depth=40, growth_rate=12, num_classes=10)
    opt['loss'] = W_BCEWithLogitsLoss()
    opt['optimizer'] = torch.optim.Adam(model.parameters(), lr=1e-5)
    opt['header'] = 'chestxray_' + enc
    opt['use_gpu'] = True

    return opt
