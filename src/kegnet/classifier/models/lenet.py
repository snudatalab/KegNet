"""
https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py
"""
import torch
from torch import Tensor
from torch import nn

from kegnet.utils.tucker import DecomposedConv2d


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.max_pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.max_pool2(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

    def compress(self, target: tuple, rank: tuple = None, hooi: bool = False):
        if 1 in target:
            self.conv1 = DecomposedConv2d(self.conv1, rank, hooi)
        if 2 in target:
            self.conv2 = DecomposedConv2d(self.conv2, rank, hooi)
        if 3 in target:
            self.conv3 = DecomposedConv2d(self.conv3, rank, hooi)
