import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--modelroot',  default='cifar10', help='Path for model file')
parser.add_argument('--modelfile',  default='netG_epoch_116.pth', help='Model file name')
parser.add_argument('--imagenum',   default=10240, type=int, help='Number of generate images')
parser.add_argument('--nrow',       default=100, type=int, help='Number of images displayed in each row of the grid')
parser.add_argument('--nz',         default=100, type=int, help='Size of the latent z vector')
parser.add_argument('--ngf',        default=64, type=int)
parser.add_argument('--outf',       default='generate_image', help='Folder to output images')
parser.add_argument('--imagefile',  default='cifar10.png', help='Output filename')
parser.add_argument('--manualSeed', default=50, type=int, help='manual seed')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(args.modelroot, args.modelfile)
out_path = os.path.join(args.outf, args.imagefile)

nz = int(args.nz)
ngf = int(args.ngf)
nc = 3

netG = Generator(nz, ngf, nc).to(device)
if os.path.exists(model_path):
    netG.load_state_dict(torch.load(model_path))
print(netG)

fixed_noise = torch.randn(args.imagenum, nz, 1, 1, device=device)
image = netG(fixed_noise)
vutils.save_image(image.detach(), out_path, nrow=args.nrow, normalize=True)

