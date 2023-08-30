# distance.py
#
# Author: Adam Sandler
# Date: 7/22/22

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import cosine_similarity
from usps_data import *

batch_size = 10**9
dist = 'euclid'
data = 'Aug2'

# get data loaders
loader1 = get_test_loader(batch_size=batch_size)
if data == 'Aug1':
    loader2 = get_test_loader(batch_size=batch_size, augment=True)[0]
elif data == 'Aug2':
    loader2 = get_test_loader(batch_size=batch_size, augment=True)[1]
elif data == 'MNIST':
    loader2 = get_mnist_loader(batch_size=batch_size)
elif data == 'GAN':
    loader2 = get_gan_loader(batch_size=batch_size, file='gan_usps.pt')
elif data == 'GAN2':
    loader2 = get_gan_loader(batch_size=batch_size, file='cgan_usps.pt')
elif 'constructed' in data:
    loader2 = get_gan_loader(batch_size=batch_size, file=data + '.pt')
else:
    raise Exception('Data not supported.')

# format input data
data1 = iter(loader1).next()
i1, _ = data1
i1 = i1.view(-1, 16**2)

data2 = iter(loader2).next()
i2, _ = data2
i2 = i2.detach().view(-1, 16**2)

# compute euclidean distances between samples
if dist == 'euclid':
    dm = distance_matrix(i1, i2)
    dmm = np.min(dm, axis=0)
elif dist == 'cosine':
    dm = cosine_similarity(i1, i2)
    dmm = np.max(dm, axis=0)
else:
    raise Exception('Distance not supported.')

# histogram
# plt.hist(dmm, bins=20)
if dist == 'euclid':
    plt.hist(dmm, bins=range(19), density=True)
    plt.xlabel('Distance')
    plt.ylim(0, .3)
elif dist == 'cosine':
    plt.hist(dmm, bins=np.linspace(0.5, 1, 21), density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylim(0, 15)
plt.ylabel('Frequency')
plt.savefig('./plots/distance_' + dist + '_' + data + '_test.png')
plt.clf()
