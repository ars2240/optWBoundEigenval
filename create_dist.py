# create_dist.py
#
# Author: Adam Sandler
# Date: 7/25/22

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset
from usps_data import *

batch_size = 10**9
dist = 'cosine'
# bins = range(18)
bins = np.linspace(0.5, .975, 20)
zeroes = 5
minmax = False
name = 'constructed6'

# get data loaders
test_loader = get_test_loader(batch_size=batch_size)
aug1_loader = get_test_loader(batch_size=batch_size, augment=True)[0]
aug2_loader = get_test_loader(batch_size=batch_size, augment=True)[1]
# mnist_loader = get_mnist_loader(batch_size=batch_size)

# format input data
data = iter(test_loader).next()
i0, l0 = data
i0 = i0.view(-1, 16**2)

data = iter(aug1_loader).next()
i1, l1 = data
i1 = i1.detach().view(-1, 16**2)

data = iter(aug2_loader).next()
i2, l2 = data
i2 = i2.detach().view(-1, 16**2)

"""
data = iter(mnist_loader).next()
i3, l3 = data
i3 = i3.detach().view(-1, 16**2)
"""

# compute euclidean distances between samples
if dist == 'euclid':
    dm = distance_matrix(i0, i1)
    dmm1 = np.min(dm, axis=0)
    dm = distance_matrix(i0, i2)
    dmm2 = np.min(dm, axis=0)
    # dm = distance_matrix(i0, i3)
    # dmm3 = np.min(dm, axis=0)
elif dist == 'cosine':
    dm = cosine_similarity(i0, i1)
    dmm1 = np.max(dm, axis=0)
    dm = cosine_similarity(i0, i2)
    dmm2 = np.max(dm, axis=0)
    # dm = cosine_similarity(i0, i3)
    # dmm3 = np.max(dm, axis=0)
else:
    raise Exception('Distance not supported.')


new_x, new_y = [], []
nz_bins = np.random.choice(bins, len(bins)-zeroes, replace=False)
# nz_bins = [2, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
# nz_bins = [0, 1, 2, 3, 4, 7, 10, 13, 14, 15, 16, 17]
step = 0.025 if dist == 'cosine' else 1
nz_bins.sort()
print(nz_bins)
k = 0
for i in nz_bins:
    if minmax:
        rows1 = np.where(np.logical_and(i <= dmm1, dmm1 < i + step))
        rows2 = np.where(np.logical_and(i <= dmm2, dmm2 < i + step))

        # min
        if k % 2 == 0:
            if len(rows1[0]) < len(rows2[0]):
                nx, ny = i1[rows1, :], l1[rows1]
                print(str(i) + ': 0')
            else:
                nx, ny = i2[rows2, :], l2[rows2]
                print(str(i) + ': 1')
        # max
        else:
            if len(rows1[0]) > len(rows2[0]):
                nx, ny = i1[rows1, :], l1[rows1]
                print(str(i) + ': 0')
            else:
                nx, ny = i2[rows2, :], l2[rows2]
                print(str(i) + ': 1')
    else:
        ds = np.random.choice(2)
        print(str(i) + ': ' + str(ds))

        if ds == 0:
            rows = np.where(np.logical_and(i <= dmm1, dmm1 < i + step))
            nx, ny = i1[rows, :], l1[rows]
        # elif ds == 1:
        else:
            rows = np.where(np.logical_and(i <= dmm2, dmm2 < i + step))
            nx, ny = i2[rows, :], l2[rows]
        """
        else:
            rows = np.where(np.logical_and(i <= dmm3, dmm3 < i + step))
            nx, ny = i3[rows, :], l3[rows]
        """

    new_x.append(nx)
    new_y.append(ny)

    k += 1

new_x, new_y = torch.cat(new_x, dim=1).view(-1, 16**2), torch.cat(new_y)
dataset = TensorDataset(new_x.view(-1, 1, 16, 16), new_y)
torch.save(dataset, './data/' + name + '.pt')

if dist == 'euclid':
    dm = distance_matrix(i0, new_x)
    dmm = np.min(dm, axis=0)
elif dist == 'cosine':
    dm = cosine_similarity(i0, new_x)
    dmm = np.max(dm, axis=0)
else:
    raise Exception('Distance not supported.')

# histogram
h_bins = np.append(bins, np.max(bins) + step)
plt.hist(dmm, bins=h_bins)
plt.savefig('./plots/distance_' + dist + '_' + name + '_test.png')
plt.clf()
