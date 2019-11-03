# main.py
#
# Author: Adam Sandler
# Date: 11/1/19
#
# Computes optimal neural network using spectral radius regularization
# Takes input parameter file
#
# Dependencies:
#   Packages: random, numpy, torch, requests, gzip, shutil, pandas
#   Files: opt

from opt import main
main('usps_CNN_mu0_1_K2')
