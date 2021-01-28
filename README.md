# optWBoundEigenval
Optimization with Bounded Eigenvalues

Author: Adam Sandler
Date: 1/28/21

## Instructions

  1. Configure an appropriate parameter file (examples in /params/)
  2. Run main.py, using parameter file as argument.

## Files/Folders

- /params/- contains parameter files used in main.py
- asymmetric_valley.py - Asymmetric Valley optimizer (modified from https://github.com/962086838/code-for-Asymmetric-Valley)
- cifar100_data.py- data loaders for CIFAR-100 data
- cifar10_data.py- data loaders for CIFAR-10 data
- cmd.py- used for GPU tracking (from https://github.com/petronny/nvsmi)
- cov_shift_acc_comp.R- compares slopes of accuracy vs. L1-norm of covariate shifts
- cov_shift_plots.R- generates plots of accuracy vs. L1-norm of covariate shifts
- cov_shift_test.py- test model on covariate shifted features
- dcnn.py- modified data loaders & neural networks (NN) for Chest X-Ray data
- dnet.py- modified DenseNet implementations
- densenet.py- DenseNet implementation (from https://github.com/andreasveit/densenet-pytorch)
- forest_data.py- forest cover type data loaders and model
- hessTest.m- computes gradient, hessian-vector product, and vector-grad hessian-vector product for example directly
- hessTest.py- computes gradient, R-Op, and R^2-Op for example and compares to MATLAB results
- kfac.py- K-Fac optimizer (from https://github.com/alecwangcq/KFAC-Pytorch)
- main.py- main executable, takes paramater file as argument
- opt.py- methods & classes for optimizing NNs with spectral radius regulation
- optim.py- Entropy-SGD optimizer (from https://github.com/ucla-vision/entropy-sgd)
- rop.py- method of R-Op and R^2-Op for test example
- usps_data.py- data loaders for USPS data

## Train/Validation/Test Splits

- USPS splits the training data into training and validation (where validation is 1/7 of the original training set), with seed 1226.
- Forest cover type splits the training data into training and validation (where validation is 1/5 of the original training set), with seed 1226.
- Chest X-ray uses the given splits.
