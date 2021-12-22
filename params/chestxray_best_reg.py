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


def options():
    # create options dictionary and some parameters
    opt = {'seed': 1226, 'tol': 0.001, 'mu': 1e-4, 'K': 0}
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

    # """
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_set = CheXpert_Dataset(use='validation', transform=transform)
    opt['test_loader'].append(test_set)
    # """
    test_set = MIMICCXR_Dataset(use='validation', transform=transform)
    opt['test_loader'].append(test_set)
    test_set = CheXpert_Dataset(use='train', transform=transform)
    opt['test_loader'].append(test_set)
    test_set = MIMICCXR_Dataset(use='train', transform=transform)
    opt['test_loader'].append(test_set)
    # """

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

    # learning rate
    def alpha(k):
        return np.exp(-k/10)

    # Training Setup
    opt['model'] = model
    opt['loss'] = W_BCEWithLogitsLoss()
    opt['optimizer'] = torch.optim.Adam(opt['model'].parameters(), lr=1e-5)
    opt['header'] = 'chestxray_E-6_' + enc
    opt['use_gpu'] = True
    opt['pow_iter'] = True
    opt['test_func'] = 'accauc sigmoid'
    opt['max_iter'] = 1
    opt['max_pow_iter'] = 100
    opt['ignore_bad_vals'] = False
    opt['pow_iter_eps'] = 0.1
    # opt['pow_iter_alpha'] = alpha
    opt['verbose'] = True
    opt['mem_track'] = True
    opt['train'] = False
    opt['test'] = False
    opt['comp_test'] = False
    opt['rho_test'] = False
    # opt['other_classes'] = list(range(1, 7))
    opt['saliency'] = 0
    opt['jaccard'] = True
    opt['comp_fname'] = './models/m-25012018-123527.pth.tar'
    # opt['fname'] = './models/m-25012018-123527.pth.tar'
    # opt['fname'] = './models/chestxray_dens121_Adam_mu' + str(opt['mu']) + '_K' + str(opt['K']) + '_trained_model_best.pt'

    return opt
