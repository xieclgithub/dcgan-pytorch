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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   default='cifar10', help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot',  default='data', help='path to dataset')
parser.add_argument('--workers',   default=8, type=int, help='number of data loading workers')
parser.add_argument('--batchSize', default=128, type=int, help='input batch size')
parser.add_argument('--imageSize', default=64, type=int, help='the height / width of the input image to network')
parser.add_argument('--nz',        default=100, type=int, help='size of the latent z vector')
parser.add_argument('--ngf',       default=64, type=int)
parser.add_argument('--ndf',       default=64, type=int)
parser.add_argument('--epochs',    default=500, type=int, help='number of epochs to train for')
parser.add_argument('--lr',        default=0.0002, type=float, help='learning rate, default=0.0002')
parser.add_argument('--beta1',     default=0.5, type=float, help='beta1 for adam. default=0.5')
parser.add_argument('--netG',      default='', help="path to netG (to continue training)")
parser.add_argument('--netD',      default='', help="path to netD (to continue training)")
parser.add_argument('--nrow',      default=16, type=int, help='Number of images displayed in each row of the grid')
parser.add_argument('--outf',      default='cifarotherseed', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed',default=80, type=int, help='manual seed')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
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
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = Discriminator(nc, ndf).to(device)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

for epoch in range(1, args.epochs+1):
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
              % (epoch, args.epochs, i, len(dataloader),
                 lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

    vutils.save_image(real, '%s/real_samples.png' % args.outf,
                      nrow=args.nrow, normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                      nrow=args.nrow, normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
