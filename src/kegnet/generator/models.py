import abc

import torch
from torch import Tensor
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_classes = None
        self.num_noises = None

    @abc.abstractmethod
    def forward(self, labels: Tensor, noises: Tensor, adjust: bool):
        pass


class DenseGenerator(Generator):
    def __init__(self,
                 num_noises: int,
                 num_classes: int,
                 num_features: int,
                 n_layers: int = 1):
        super().__init__()
        units = 120
        self.num_classes = num_classes
        self.num_noises = num_noises

        layers = [nn.Linear(num_noises + num_classes, units),
                  nn.ELU(),
                  nn.BatchNorm1d(units)]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(units, units),
                nn.ELU(),
                nn.BatchNorm1d(units)])

        layers.append(nn.Linear(units, num_features))
        self.layers = nn.Sequential(*layers)
        self.adjust = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, labels: Tensor, noises: Tensor, adjust: bool = True) -> Tensor:
        out = self.layers(torch.cat((noises, labels), dim=1))
        if adjust:
            out = self.adjust(out)
        return out


class ImageGenerator(Generator):
    def __init__(self, num_noises: int, num_classes: int, num_channels: int):
        super(ImageGenerator, self).__init__()

        fc_nodes = [num_noises + num_classes, 256, 128]
        cv_nodes = [fc_nodes[-1], 64, 16, 4, num_channels]

        self.num_classes = num_classes
        self.num_noises = num_noises
        self.fc = nn.Sequential(
            nn.Linear(fc_nodes[0], fc_nodes[1]),
            nn.BatchNorm1d(fc_nodes[1]),
            nn.ReLU(),
            nn.Linear(fc_nodes[1], fc_nodes[2]),
            nn.BatchNorm1d(fc_nodes[2]),
            nn.ReLU())

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(cv_nodes[0], cv_nodes[1], 4, 2, 0, bias=False),
            nn.BatchNorm2d(cv_nodes[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[1], cv_nodes[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cv_nodes[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[2], cv_nodes[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cv_nodes[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[3], cv_nodes[4], 4, 2, 1, bias=False),
            nn.Tanh())

    @staticmethod
    def _normalize_images(layer: Tensor) -> Tensor:
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

    def forward(self, labels: Tensor, noises: Tensor, adjust: bool = True) -> Tensor:
        out = self.fc(torch.cat((noises, labels), dim=1))
        out = self.conv(out.view((out.size(0), out.size(1), 1, 1)))
        if adjust:
            out = self._normalize_images(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_features: int, out_targets: int, n_layers: int):
        super(Decoder, self).__init__()

        layers = [nn.Linear(in_features, 120),
                  nn.ELU(),
                  nn.BatchNorm1d(120)]

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(120, 120),
                nn.ELU(),
                nn.BatchNorm1d(120)])

        layers.append(nn.Linear(120, out_targets))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        out = x.view((x.size(0), -1))
        out = self.layers(out)
        return out
