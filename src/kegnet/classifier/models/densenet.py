"""
https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as func

from kegnet.utils.tucker import DecomposedConv2d


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(func.relu(self.bn1(x)))
        out = self.conv2(func.relu(self.bn2(out)))
        out = torch.cat((out, x), 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(func.relu(self.bn(x)))
        out = func.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, num_classes, num_channels, growth_rate, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(num_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = func.avg_pool2d(func.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def compress(self, hooi: bool = False):
        tucker_func = DecomposedConv2d

        for layers in (self.dense1, self.dense2, self.dense3, self.dense4):
            for block in layers:
                block.conv1 = tucker_func(block.conv1, None, hooi)
                block.conv2 = tucker_func(block.conv2, None, hooi)


def densenet121(growth_rate: int = 32, **kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=growth_rate, **kwargs)


def densenet169(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, **kwargs)


def densenet201(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, **kwargs)


def densenet161(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, **kwargs)
