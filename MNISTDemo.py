import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import Module
import Function

os.makedirs("images/cgan/", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--Gchannels", type=int, default=128, help="start_channels_for_G")
parser.add_argument("--n_classes", type=int, default=5, help="num of class of data (labels)")
opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Configure data loader
os.makedirs("Datas/", exist_ok=True)
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "Datas/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


def cgan_demo():
    generator = Module.GeneratorCGAN(opt.latent_dim, opt.n_classes, img_shape)  # latent_dim should be 100
    discriminator = Module.DiscriminatorCGAN(opt.n_classes, img_shape)

    Function.train_cgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                        opt.latent_dim, opt.n_classes, cuda, fist_train=False)


def cdcgan_demo():
    generator = Module.GeneratorCDCGAN(opt.latent_dim, opt.n_classes, img_shape)  # latent_dim should be 20
    discriminator = Module.DiscriminatorCDCGAN(opt.n_classes, img_shape, opt.latent_dim)

    Function.train_cdcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                          opt.latent_dim, opt.n_classes, cuda,
                          fist_train=True)


def cgan_new_demo():
    show_data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "Datas/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=100,
        shuffle=True,
    )
    generator = Module.GeneratorCGANNew(img_shape)
    discriminator = Module.DiscriminatorCGANNew(img_shape)

    Function.train_cgan_new(generator, discriminator, data_loader,show_data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                            cuda, first_train=True)


def dcwcgan_demo():
    generator = Module.GeneratorDCWCGAN(opt.latent_dim, opt.n_classes, img_shape)  # latent_dim should be 100
    discriminator = Module.DiscriminatorDCWCGAN(opt.n_classes, img_shape)

    Function.train_dcwcgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                           opt.latent_dim, opt.n_classes, cuda,
                           fist_train=True)


def main():
    mode = 3
    if mode == 0:
        cgan_demo()
    elif mode == 1:
        cdcgan_demo()
    elif mode == 3:
        cgan_new_demo()
    else:
        print("mode error")


if __name__ == '__main__':
    main()
