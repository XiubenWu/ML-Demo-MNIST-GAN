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


class GeneratorCDCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super(GeneratorCDCGAN, self).__init__()

        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, 100)
        self.l1 = nn.Linear(latent_dim + 100, img_shape[-1] ** 2)

        self.channel1 = 32
        self.channel2 = 64
        self.channel3 = 64

        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.img_shape[0]),

            nn.Conv2d(self.img_shape[0], self.channel1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel1, self.channel2, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel2, self.channel3, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel3, self.img_shape[0], 3, stride=1, padding=1),
        )

    def forward(self, noise, labels):
        em_labels = self.label_emb(labels)
        inputs = torch.cat((em_labels, noise), -1)
        inputs = self.l1(inputs)
        out = inputs.view(inputs.size(0), *self.img_shape)
        img = self.conv(out)
        return img


class DiscriminatorCDCGAN(nn.Module):
    def __init__(self, n_classes, img_shape, latent_dim):
        super(DiscriminatorCDCGAN, self).__init__()

        self.img_shape = img_shape
        self.em_label = nn.Embedding(n_classes, latent_dim)
        self.l1 = nn.Linear(latent_dim + img_shape[1] * img_shape[2], 256)
        self.l2 = nn.Linear(256, img_shape[1] * img_shape[2])

        self.channel1 = 32
        self.channel2 = 64
        self.channel3 = 16
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0], self.channel1, 3, 1, 0),  # 32 to 30
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 30 to 15

            nn.Conv2d(self.channel1, self.channel2, 4, 1, 0),  # 15 to 12
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2),  # 12 to 6

            nn.Conv2d(self.channel2, self.channel3, 3, 1, 1),

        )
        self.l3 = nn.Sequential(
            nn.Linear(6 * 6 * self.channel3, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, labels):
        inputs = torch.cat((self.em_label(labels), inputs.view(inputs.size(0), -1)), -1)
        inputs = self.l1(inputs)
        inputs = self.l2(inputs)
        inputs = inputs.view(inputs.size(0), *self.img_shape)
        inputs = self.conv(inputs)
        out = inputs.view(inputs.shape[0], -1)
        valid = self.l3(out)

        return valid


class GeneratorDCWCGAN(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        '''
        :param latent_dim: length of noise  opt.latent_dim
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorDCWCGAN, self).__init__()

        feature_dim = 20

        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.schannels = 8
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, feature_dim)

        self.init_size = img_shape[-1] // 4  # upsample * 2
        self.l1 = nn.Sequential(nn.Linear(latent_dim + feature_dim, self.schannels * self.init_size ** 2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(self.schannels),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.schannels, channel1, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(channel1, channel2, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel2, channel3, 3, stride=1, padding=1),
            nn.BatchNorm2d(channel3, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel3, img_shape[0], 3, stride=1, padding=1),
        )

    def forward(self, noise, labels):
        '''
        :param noise:
        :param labels:
        :return: (btach size,channels,image size,image size)
        '''
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        gen_input = self.l1(gen_input)
        gen_input = gen_input.view(gen_input.size(0), self.schannels, self.init_size, self.init_size)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class DiscriminatorDCWCGAN(nn.Module):
    def __init__(self, n_classes, img_shape):
        '''
        :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(DiscriminatorDCWCGAN, self).__init__()

        feature_dim = 20
        channel1 = 64
        channel2 = 128
        channel3 = 256

        self.img_shape = img_shape
        self.em_label = nn.Embedding(n_classes, feature_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, channel1, 3, 1, 0),  # 32* to 30*
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
            nn.Linear(6 * 6 * channel3 + feature_dim, 100),
            nn.Linear(100, 1)

        )

    def forward(self, inputs, labels):
        emb_input = self.em_label(labels)
        img_input = self.conv(inputs)
        img_input = img_input.view(img_input.size(0), -1)  # [img_input.size(0),6*6*channel2]
        inputs = torch.cat((emb_input, img_input), -1)
        valid = self.l1(inputs)

        return valid


class GeneratorCGANNew(nn.Module):
    def __init__(self, img_shape):
        '''
        # :param latent_dim: length of noise  opt.latent_dim
        # :param n_classes: num of class of data (labels)  opt.n_classes
        :param img_shape: turtle (channels,img size,img size)
        note:feature_dim must be changed in both of G and D
        '''
        super(GeneratorCGANNew, self).__init__()

        channel0 = 4
        channel1 = 16
        channel2 = 32
        channel3 = 64
        channel4 = 128

        self.img_shape = img_shape

        self.extend = nn.Conv2d(1, channel0, 3, 1, 1)

        self.convDown = nn.Sequential(
            nn.Conv2d(channel0 + 1, channel1, 3, 1, 1),
            nn.BatchNorm2d(channel1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )
        self.convUp = nn.Sequential(
            nn.Conv2d(channel1, channel2, 3, 1, 1),
            nn.BatchNorm2d(channel2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2)
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
        x = self.extend(inputs)
        x, mean, std = self.PONO(x)
        noise = torch.randn(inputs.shape[0], 1, self.img_shape[1], self.img_shape[2]).cuda()
        x = torch.cat((x, noise), dim=1)
        x = self.convDown(x)
        x = self.convUp(x)
        x = self.MS(x, mean, std)
        imgs = self.convNormal(x)
        return imgs

    def PONO(self, x, eps=0.00001):
        mean = x.mean(dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + eps).sqrt()
        x = (x - mean) / std
        return x, mean, std

    def MS(self, x, mean, std):
        '''Decoding
        :param x: inputs
        :param mean: mean
        :param std: std
        :return: processed x
        '''
        x.mul_(std)
        x.add_(mean)
        return x


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