# optWBoundEigenval
Optimization with Bounded Eigenvalues

Author: Adam Sandler
Date: 1/12/21

## Instructions

  1. Configure an appropriate parameter file (examples in /params/)
  2. Run main.py, using parameter file as argument.

## Files/Folders

- /params/- contains parameter files used in main.py
- cifar100_data.py- data loaders for CIFAR-100 data
- cifar10_data.py- data loaders for CIFAR-10 data
- cmd.py- used for GPU tracking (from https://github.com/petronny/nvsmi)
- dcnn.py- modified data loaders & neural networks (NN) for Chest X-Ray data
- densenet.py- DenseNet implementation (from https://github.com/andreasveit/densenet-pytorch)
- hessTest.m- computes gradient, hessian-vector product, and vector-grad hessian-vector product for example directly
- hessTest.py- computes gradient, R-Op, and R^2-Op for example and compares to MATLAB results
- main.py- main executable, takes paramater file as argument
- opt.py- methods & classes for optimizing NNs with spectral radius regulation
- rop.py- method of R-Op and R^2-Op for test example
- usps_data.py- data loaders for USPS data
