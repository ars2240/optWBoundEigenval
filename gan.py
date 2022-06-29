# Modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
# Paper: https://arxiv.org/abs/1411.1784

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=2e-5, help="adam: weight decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--gen_images", type=int, default=10000, help="number of generated images")
parser.add_argument("--nodes", type=int, default=32, help="number of nodes in the 1st layer of the network")
parser.add_argument("--train", type=int, default=True, help="whether or not to train model")
parser.add_argument("--scheduler", type=int, default=True, help="whether or not to use learning rate scheduler")
parser.add_argument("--cos", type=int, default=True, help="whether or not to use cosine annealing lr")
parser.add_argument("--rand", type=float, default=0.3, help="amount to randomly fudge labels")
parser.add_argument("--swap", type=float, default=0.01, help="probability of swapping labels")
parser.add_argument("--d_iter", type=int, default=1, help="number of discriminator iterations per generator")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, n=128):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, n, normalize=False),
            *block(n, n*2),
            *block(n*2, n*4),
            *block(n*4, n*8),
            nn.Linear(n*8, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, n=128):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), n*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n*4, n*4),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n*4, n*4),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n*4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(n=opt.nodes)
discriminator = Discriminator(n=opt.nodes)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
trans = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize(
    mean=[0.24510024090766908], std=[0.29806136205673217],)])
dataloader = torch.utils.data.DataLoader(
    datasets.USPS("./data", train=True, download=True, transform=trans),
    batch_size=opt.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.USPS("./data", train=False, download=True, transform=trans),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

# learning rate scheduler
if opt.cos:
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=opt.n_epochs)
else:
    def beta(k):
        return 1/(k+1)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=beta)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=beta)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

if opt.train:
    print('Epoch\tD loss\tG loss')
    for epoch in range(opt.n_epochs):
        g_loss_list, d_loss_list, size = [], [], []
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            # """
            if opt.rand > 0:
                valid = Variable(torch.from_numpy(np.random.uniform(1.0 - opt.rand, 1.0, size=(batch_size, 1))).float(),
                                 requires_grad=False)
                fake = Variable(torch.from_numpy(np.random.uniform(0.0, opt.rand, size=(batch_size, 1))).float(),
                                requires_grad=False)
            else:
                valid = Variable(FloatTensor(batch_size, 1).fill_(1), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0), requires_grad=False)

            if opt.swap > np.random.uniform():
                valid, fake = fake, valid

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for _ in range(opt.d_iter):
                optimizer_D.zero_grad()

                # Loss for real images
                validity_real = discriminator(real_imgs, labels)
                d_real_loss = adversarial_loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = adversarial_loss(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

            g_loss_list.append(g_loss.item())
            d_loss_list.append(d_loss.item())
            size.append(batch_size)

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done)

        print("%d\t%f\t%f" % (epoch, np.average(d_loss_list, weights=size), np.average(g_loss_list, weights=size)))

        # step learning rate schedulers
        if opt.scheduler:
            scheduler_G.step()
            scheduler_D.step()

    torch.save(generator.state_dict(), './models/gan_generator.pt')
    torch.save(discriminator.state_dict(), './models/gan_discriminator.pt')
else:
    generator.load_state_dict(torch.load('./models/gan_generator.pt'))
    discriminator.load_state_dict(torch.load('./models/gan_discriminator.pt'))

# compute test accuracy
size, real_correct, fake_correct = 0, 0, 0
for i, (imgs, labels) in enumerate(testloader):

    batch_size = imgs.shape[0]

    # Adversarial ground truths
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(imgs.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))

    # Sample noise and labels as generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
    gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

    # Generate a batch of images
    gen_imgs = generator(z, gen_labels)

    # Loss measures generator's ability to fool the discriminator
    validity_real = discriminator(real_imgs.detach(), labels)
    validity_fake = discriminator(gen_imgs.detach(), gen_labels)

    real_correct += (validity_real > .5).sum()
    fake_correct += (validity_fake < .5).sum()
    size += batch_size

print('Test Accuracy: %f' % ((real_correct+fake_correct)/(2*size)))
print('Real Accuracy: %f' % (real_correct/size))
print('Fake Accuracy: %f' % (fake_correct/size))

# Generate Images

# Sample noise and labels as generator input
z = Variable(FloatTensor(np.random.normal(0, 1, (opt.gen_images, opt.latent_dim))))
gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.gen_images)))

# Generate a batch of images
gen_imgs = generator(z, gen_labels)

dataset = TensorDataset(gen_imgs, gen_labels)

torch.save(dataset, './data/gan_usps.pt')
