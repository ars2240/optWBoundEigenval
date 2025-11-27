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

import sys
from opt import main

if len(sys.argv) == 2:
    main(sys.argv[1])
else:
    main('usps_CNN_lobpcg')
    main('forest_lobpcg')
