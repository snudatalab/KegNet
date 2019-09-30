"""
https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
"""
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor

from kegnet.utils import tucker


class IdentityMapping(nn.Module):
    def __init__(self, num_filters: int, channels_in: int, stride: int):
        super(IdentityMapping, self).__init__()
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = func.pad(x, [0, 0, 0, 0, 0, self.num_zeros])
        out = self.identity(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_filters: int, channels_in: int = None, stride: int = 1):
        super(ResBlock, self).__init__()

        if channels_in is None or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            self.projection = IdentityMapping(num_filters, channels_in, stride)

        self.conv1 = nn.Conv2d(channels_in, num_filters, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection:
            residual = self.projection(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, n: int = 9):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 16, 16, 1)
        self.layers2 = self._make_layer(n, 32, 16, 2)
        self.layers3 = self._make_layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(n: int, num_filters: int, channels_in: int, stride: int):
        layers = [ResBlock(num_filters, channels_in, stride)]
        for _ in range(1, n):
            layers.append(ResBlock(num_filters))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def compress(self, target: tuple, rank: object, hooi: bool):
        for layer in self.layers3:
            if 1 in target:
                layer.conv1 = tucker.DecomposedConv2d(layer.conv1, rank, hooi)
            if 2 in target:
                layer.conv2 = tucker.DecomposedConv2d(layer.conv2, rank, hooi)


def resnet14(**kwargs) -> ResNet:
    return ResNet(n=2, **kwargs)


def resnet20(**kwargs) -> ResNet:
    return ResNet(n=3, **kwargs)


def resnet56(**kwargs) -> ResNet:
    return ResNet(n=9, **kwargs)
