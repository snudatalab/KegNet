"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
from torch import nn


class DenseGenerator(nn.Module):
    """
    Generator for unstructured datasets.
    """

    def __init__(self, num_classes, num_features, num_noises=10, units=120,
                 n_layers=1):
        """
        Class initializer.
        """
        super().__init__()
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

    def forward(self, labels, noises, adjust=True):
        """
        Forward propagation.
        """
        out = self.layers(torch.cat((noises, labels), dim=1))
        if adjust:
            out = self.adjust(out)
        return out


class ImageGenerator(nn.Module):
    """
    Generator for image datasets.
    """

    def __init__(self, num_classes, num_channels, num_noises=10):
        """
        Class initializer.
        """
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
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

    def forward(self, labels, noises, adjust=True):
        """
        Forward propagation.
        """
        out = self.fc(torch.cat((noises, labels), dim=1))
        out = self.conv(out.view((out.size(0), out.size(1), 1, 1)))
        if adjust:
            out = self.normalize_images(out)
        return out


class Decoder(nn.Module):
    """
    Decoder for both unstructured and image datasets.
    """

    def __init__(self, in_features, out_targets, n_layers, units=120):
        """
        Class initializer.
        """
        super(Decoder, self).__init__()

        layers = [nn.Linear(in_features, units),
                  nn.ELU(),
                  nn.BatchNorm1d(units)]

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(units, units),
                nn.ELU(),
                nn.BatchNorm1d(units)])

        layers.append(nn.Linear(units, out_targets))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = x.view((x.size(0), -1))
        out = self.layers(out)
        return out
