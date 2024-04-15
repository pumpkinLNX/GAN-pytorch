# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transform
import torchvision.utils as vutils

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import display

import argparse
import os
import random

manualseed = 999
print("Manual seed:", manualseed)
random.seed(manualseed)
torch.manual_seed(manualseed)

# Root directory for dataset
dataroot = "D:/lnx/code/by myself/gan-cnn/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

dataset=dset.ImageFolder(root=dataroot,
                         transform=transform.Compose([
                             transform.Resize(image_size),
                             transform.CenterCrop(image_size),
                             transform.ToTensor(),
                             transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device=torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

real_batch=next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# 显示训练数据图像
#plt.show()

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu=ngpu
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create a Generator
netG=Generator(ngpu).to(device)
if (device.type=='cuda') and (ngpu>1):
    netG=nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
#######
#print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu=ngpu
        self.main=nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create a Discriminator
netD=Discriminator(ngpu).to(device)
if (device.type=='cuda') and (ngpu>1):
    netD=nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
#####
#print(netD)

criterion=nn.BCELoss()
fixed_noise=torch.randn(64, nz, 1, 1, device=device)
real_label=1.
fake_label=0.

optimizerD=optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG=optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#############################################
# training loop
img_list=[]
G_losses=[]
D_losses=[]
iters=0
#####
horaxis=[]

print("Start training loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ### Update D network
        # 1. train with real data
        netD.zero_grad()
        real_cpu=data[0].to(device)
        b_size=real_cpu.size(0)
        label=torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output=netD(real_cpu).view(-1)

        errD_real=criterion(output, label)
        errD_real.backward()
        D_x=output.mean().item()

        # 2. train with fake data
        noise=torch.randn(b_size, nz, 1, 1, device=device)
        fake=netG(noise)
        label.fill_(fake_label)
        output=netD(fake.detach()).view(-1)

        errD_fake=criterion(output, label)
        errD_fake.backward()
        D_G_z1=output.mean().item()

        errD=errD_real+ errD_fake
        optimizerD.step()

        ########### Update G network
        netG.zero_grad()
        label.fill_(real_label)
        output=netD(fake).view(-1)

        errG=criterion(output, label)
        errG.backward()
        D_G_z2=output.mean().item()

        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

       # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        #########################
        x=iters+1;
        horaxis.append(x)
        # if (x % 10 == 0):
        #     plt.figure()
        #     plt.plot(horaxis, G_losses, 'b')
        #     plt.plot(horaxis, D_losses, 'r')
        #     plt.show()
        #     plt.close()


        iters+=1

###################plot images##########################
real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("real images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("fake images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
























































