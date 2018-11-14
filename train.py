from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import Generator, Discriminator
from get_task import get_task

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', default='workdir', help='working dir')
parser.add_argument('--dataroot', default='data', help='dataroot')
parser.add_argument('--task', default=0, type=int, help='number for task')

args = parser.parse_args()
print(args)

file_path = os.path.join(args.workdir, str(args.task))

if not os.path.exists(file_path):
    raise ValueError("directory %s is not find" % file_path)

config = get_task(file_path)

try:
    os.mkdir(config.outf)
except OSError:
    pass

if config.manual_seed is None:
    config.manual_seed = random.randint(1, 10000)
print("Random Seed: ", config.manual_seed)
random.seed(config.manual_seed)
torch.manual_seed(config.manual_seed)

cudnn.benchmark = True

if config.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(config.imageSize),
                                   transforms.CenterCrop(config.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif config.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(config.imageSize),
                            transforms.CenterCrop(config.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif config.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif config.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, config.image_size, config.imagesize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                         shuffle=True, num_workers=int(config.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nz = int(config.nz)
ngf = int(config.ngf)
ndf = int(config.ndf)
nc = 3

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = Generator(nz, ngf, nc).to(device)
netG.apply(weights_init)
if config.netG != '':
    netG.load_state_dict(torch.load(config.netG))
print(netG)

netD = Discriminator(nc, ndf).to(device)
netD.apply(weights_init)
if config.netD != '':
    netD.load_state_dict(torch.load(config.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(config.batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

for epoch in range(1, config.epochs+1):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = lossD_real + lossD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        lossG = criterion(output, label)
        lossG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, config.epochs, i, len(dataloader),
                 lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

    vutils.save_image(config, '%s/real_samples.png' % config.outf,
                      nrow=config.nrow, normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (config.outf, epoch),
                      nrow=config.nrow, normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (config.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (config.outf, epoch))
