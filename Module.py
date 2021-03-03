import numpy as np
import torch

import torch.nn as nn


class GeneratorCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        '''
        super(GeneratorCGAN, self).__init__()

        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
        )

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        '''
        super(DiscriminatorCGAN, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_input)
        return validity


class GeneratorCGANNew(nn.Module):
    def __init__(self, img_shape):
        '''
        # :param latent_dim: length of noise  opt.latent_dim
        # :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorCGANNew, self).__init__()

        channel0 = 1
        channel1 = 16
        channel2 = 32
        channel3 = 64
        channel4 = 128

        self.img_shape = img_shape

        self.convDown = nn.Sequential(
            nn.Conv2d(1, channel1, 3, 1, 1),
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(channel1, channel0, 3, 1, 1)
        )
        self.convUp = nn.Sequential(
            nn.Conv2d(channel0 + 1, channel2, 3, 1, 1),
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=4)
        )

        self.convNormal = nn.Sequential(
            nn.Conv2d(channel2, channel3, 3, 1, 1),
            nn.BatchNorm2d(channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel3, channel4, 3, 1, 1),
            nn.BatchNorm2d(channel4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel4, img_shape[0], 3, 1, 1)
        )

    def forward(self, inputs):
        x = inputs
        x = self.convDown(x)
        noise = torch.randn(x.shape[0], 1, self.img_shape[1] // 4, self.img_shape[2] // 4).cuda()
        x = torch.cat((x, noise), dim=1)
        x = self.convUp(x)
        imgs = self.convNormal(x)
        return imgs


class DiscriminatorCGANNew(nn.Module):
    def __init__(self, img_shape):
        '''
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorCGANNew, self).__init__()

        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.img_shape = img_shape
        self.conv = nn.Sequential(
            nn.Conv2d(self.img_shape[0], channel1, 3, 1, 0),  # 32* to 30*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30* to 15*

            nn.Conv2d(channel1, channel2, 4, 1, 0),  # 15* to 12*
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12* to 6*

            nn.Conv2d(channel2, channel3, 3, 1, 1),  # 6* to 6*
        )
        self.l1 = nn.Sequential(
            nn.Linear(6 * 6 * channel3, 100),
            nn.Linear(100, 1)

        )

    def forward(self, inputs):
        img_input = self.conv(inputs)
        img_input = img_input.view(img_input.size(0), -1)
        valid = self.l1(img_input)
        return valid
