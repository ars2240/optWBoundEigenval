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

import os
import numpy as np
import sys
import torch
#sys.path.insert(0, '/home/hddraid/shared_data/chest_xray8/code/VClassifier/')  # add folder containing dcnn to path
from dcnn import *
from kfac import KFACOptimizer


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 0, 'K': 0}
    enc = 'dens121'  # model type

    # batch size
    batch_size = 4
    opt['batch_size'] = batch_size

    # set number of threads
    """
    nthreads = 4
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    torch.set_num_threads(nthreads)
    from multiprocessing import Pool
    Pool(nthreads-1, main())
    """

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
    opt['test_loader'] = []
    test_set = ChestXray_Dataset(use='test', transform=transform)
    opt['test_loader'].append(test_set)

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_set = CheXpert_Dataset(use='validation', transform=transform)
    opt['test_loader'].append(test_set)
    test_set = MIMICCXR_Dataset(use='validation', transform=transform)
    opt['test_loader'].append(test_set)
    test_set = CheXpert_Dataset(use='train', transform=transform)
    opt['test_loader'].append(test_set)
    test_set = MIMICCXR_Dataset(use='train', transform=transform)
    opt['test_loader'].append(test_set)

    # Create neural network
    if enc == 'alex':
        model = MyAlexNet(14)
    elif enc == 'res50':
        model = MyResNet50(14)
    elif enc == 'vgg16bn':
        model = MyVggNet16_bn(14)
    elif enc == 'dens121':
        model = MyDenseNet121(14)
    elif enc == 'dens161':
        model = MyDensNet161(14)
    elif enc == 'dens201':
        model = MyDensNet201(14)
    elif enc == 'dens121':
        model = MyDensNet121(14)

    # Training Setup
    opt['model'] = model
    opt['loss'] = W_BCEWithLogitsLoss()
    opt['optimizer'] = torch.optim.SGD(opt['model'].parameters(), lr=1e-5)
    opt['header'] = 'chestxray_AsymValley_' + enc
    opt['use_gpu'] = True
    opt['pow_iter'] = False
    opt['test_func'] = 'accauc sigmoid'
    opt['max_pow_iter'] = 100
    opt['ignore_bad_vals'] = True
    opt['pow_iter_eps'] = 0.1
    opt['verbose'] = True
    opt['train'] = True
    opt['test'] = True
    opt['comp_test'] = True
    opt['fname'] = './models/m-25012018-123527.pth.tar'
    opt['asymmetric_valley'] = True
    opt['swa_start'] = 1
    opt['sgd_start'] = 40
    opt['max_iter'] = 90

    return opt
