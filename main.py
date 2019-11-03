# main.py
#
# Author: Adam Sandler
# Date: 11/1/19
#
# Computes optimal neural network using spectral radius regularization
# Takes input parameter file
#
# Dependencies:
#   Packages: bz2, gzip, numpy, pandas, requests, shutil, sklearn, torch
#   Files: opt, ./params/file

from opt import main
main('usps_CNN_mu0_1_K2')
