import os
from torch.autograd import Variable
import torch
import numpy as np
import itertools
from torchvision.utils import save_image


def train_cgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
               fist_train=False):
    path = "GANParameters/CGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    loss = torch.nn.MSELoss()

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(data_loader) + i
            if batches_done % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )
            if batches_done % 1000 == 0:
                sample_image(generator, n_row=10, batches_done=batches_done, FloatTensor=FloatTensor,
                             LongTensor=LongTensor)
            if batches_done % 3000 == 0:
                torch.save(generator.state_dict(), path + "/generator.pt")
                torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_cdcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                 fist_train=False):
    path = "GANParameters/CDCGAN"
    os.makedirs(path, exist_ok=True)
    os.makedirs("images/cdcgan/", exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    loss = torch.nn.BCELoss()

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(data_loader) + i
            if batches_done % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )
            if batches_done % 1000 == 0:
                z = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, 100))))
                show_labels = np.array([num for _ in range(10) for num in range(10)])
                show_labels = Variable(LongTensor(show_labels))
                show_imgs = generator(z, show_labels)
                save_image(show_imgs.data, "images/cdcgan/%d.png" % batches_done, nrow=10, normalize=True)
            if batches_done % 2000 == 0:
                torch.save(generator.state_dict(), path + "/generator.pt")
                torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_dcwcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                  fist_train=False):
    path = "GANParameters/DCWCGAN"
    os.makedirs(path, exist_ok=True)
    os.makedirs("images/dcwcgan/", exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            # imgs = imgs[:, np.newaxis, :, :]
            real_imgs = imgs.type(FloatTensor)
            labels = labels.type(LongTensor)

            optimizer_D.zero_grad()

            z = FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim)))
            gen_labels = LongTensor(np.random.randint(0, n_classes, batch_size))

            fake_imgs = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_imgs, labels)) + torch.mean(discriminator(fake_imgs, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % 5 == 0:
                gen_imgs = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            batches_done = epoch * len(data_loader) + i
            if batches_done % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
            if batches_done % 1000 == 0:
                z = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, 100))))
                show_labels = np.array([num for _ in range(10) for num in range(10)])
                show_labels = Variable(LongTensor(show_labels))
                show_imgs = generator(z, show_labels)
                save_image(show_imgs.data, "images/dcwcgan/%d.png" % batches_done, nrow=10, normalize=True)
            if batches_done % 2000 == 0:
                torch.save(generator.state_dict(), path + "/generator.pt")
                torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_cgan_new(generator, discriminator, data_loader, show_data_loader, n_epochs, lr, b1, b2, cuda,
                   first_train=False):
    path = "GANParameters/CGANNew"
    os.makedirs(path, exist_ok=True)
    os.makedirs("images/cgannew/", exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Configure input
            # imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(FloatTensor))  # cuda()

            optimizer_D.zero_grad()

            fake_imgs = generator(real_imgs).detach()  # self noise

            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 5 == 0:
                gen_imgs = generator(real_imgs)

                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()
            batches_done = epoch * len(data_loader) + i
            if batches_done % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
            if batches_done % 1000 == 0:
                for i_show, (show_imgs, show_labels) in enumerate(show_data_loader):
                    index_list = []
                    for j in range(10):
                        index = (show_labels == j).nonzero()
                        if index.shape[0] == 0:
                            break
                        while index.shape[0] < 10:
                            index = torch.cat((index, index), dim=0)
                        index_list.append(index)
                    if j != 9:
                        continue
                    input_imgs = torch.empty((100, 1, 32, 32))
                    for row in range(10):
                        for col in range(10):
                            input_imgs[row * 10 + col] = show_imgs[index_list[col][row]]
                    if cuda:
                        input_imgs = input_imgs.type(FloatTensor)
                    show_imgs = generator(input_imgs)

                    save_image(show_imgs.data, "images/cgannew/%d.png" % batches_done, nrow=10, normalize=True)
                    break

            if batches_done % 2000 == 0:
                torch.save(generator.state_dict(), path + "/generator.pt")
                torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def sample_image(generator, n_row, batches_done, FloatTensor, LongTensor):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, 100))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/cgan/%d.png" % batches_done, nrow=n_row, normalize=True)
