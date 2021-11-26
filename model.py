import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable

from deform_conv import ConvOffset2D


class SENet(nn.Module):
    def __init__(self, n_channels, squeeze = 16):
        super(SENet, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_channels, n_channels//squeeze, 1),
            nn.ReLU(True),
            nn.Conv2d(n_channels//squeeze, n_channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise Exception("Unknown mode in GCT with type:{}".format(self.mode))
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=3//2)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, n_channels, atten_type = 'none'):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2),
            nn.ReLU(True),
            nn.Conv2d(n_channels, n_channels, 3, padding=3//2),
        )
        if atten_type == 'none':
            self.attention = nn.Sequential()
        elif atten_type == 'senet':
            self.attention = SENet(n_channels)
        elif atten_type == 'gct':
            self.attention = GCT(n_channels)
        else:
            raise Exception("Unknown attention type:{}".format(atten_type))
    def forward(self, x):
        y = self.conv(x)
        y = self.attention(y)
        return x + y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.alignment = nn.Sequential(
            ConvOffset2D(filters=6),
            nn.Conv2d(6, 32, 3, padding=3//2),
        )
        self.mapping = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, 32, 3, padding=3//2),
        )
        self.restoration = nn.Sequential(
            ConvReLU(32, 32),
            nn.Conv2d(32, 6, 3, padding=3//2),
        )
    def forward(self, mix):
        upper, lower = mix[:, :, 0::2, :], mix[:, :, 1::2, :]
        mixture = torch.cat((upper, lower), 1)
        align = self.alignment(mixture)
        mapping = self.mapping(align) + align
        restore = self.restoration(mapping)
        restoreA, restoreB = restore[:, 0:3, :, :], restore[:, 3:6, :, :]
        return restoreA, restoreB
