# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import argparse

import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()

# parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
# parser.add_argument('--nChannel', metavar='N', default=100, type=int,
#                     help='number of channels')
# parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
#                     help='number of maximum iterations')
# parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
#                     help='minimum number of labels')
# parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
#                     help='learning rate')
# parser.add_argument('--nConv', metavar='M', default=2, type=int,
#                     help='number of convolutional layers')
# parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
#                     help='number of superpixels')
# parser.add_argument('--compactness', metavar='C', default=100, type=float,
#                     help='compactness of superpixels')
# parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
#                     help='visualization flag')
# parser.add_argument('--input', metavar='FILENAME',
#                     help='input image file name', required=True)
# args = parser.parse_args()

args = argparse.Namespace(
    nChannel=100,
    maxIter=1000,
    minLabels=3,
    lr=0.1,
    nConv=2,
    num_superpixels=10000,
    compactness=100,
    visualize=1,
)


class USCNet(nn.Module):
    def __init__(self,input_dim):
        super(USCNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
