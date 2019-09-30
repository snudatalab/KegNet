import tensorly as tl
import tensorly.decomposition as decomp
import torch
import torch.nn as nn
from tensorly.tucker_tensor import tucker_to_tensor
from torch.nn import Module

from kegnet.utils.vbmf import EVBMF

tl.set_backend('pytorch')


class DecomposedConv2d(Module):
    """
    The most basic convolutional module with tucker-2 decomposition

    Reference
        - Compression of Deep Convolutional Neural Networks
          for Fast and Low Power Mobile Applications
    """

    def __init__(self, conv_layer: Module,
                 rank: tuple or str = 'evbmf',
                 hooi: bool = False):
        """
        Constructor for Tucker2DecomposedConv Layer

        @param conv_layer: Original layer to decompose
        @param rank: Projection rank (default None)
            If None, run EVBMF to calculate rank
        @param hooi: whether using HOOI to initialize layer weight (default F)
            If true, it is same as tucker decomposition
            Else, it initializes randomly
        """

        super(DecomposedConv2d, self).__init__()

        # get device
        device = conv_layer.weight.device

        # get weight and bias
        weight = conv_layer.weight.data

        out_channels, in_channels, _, _ = weight.shape

        if rank == 'evbmf':
            # run EVBMF and get estimated ranks
            unfold_0 = tl.base.unfold(weight, 0)
            unfold_1 = tl.base.unfold(weight, 1)
            _, diag_0, _, _ = EVBMF(unfold_0)
            _, diag_1, _, _ = EVBMF(unfold_1)
            out_rank = diag_0.shape[0]
            in_rank = diag_1.shape[1]
        elif isinstance(rank, float):
            out_rank = int(out_channels * rank)
            in_rank = int(in_channels * rank)
        elif isinstance(rank, tuple):
            in_rank, out_rank = rank
        else:
            raise ValueError(rank)
        # print('Projection Ranks: [{}, {}]'.format(in_rank, out_rank))

        # initialize layers
        self.in_channel_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=conv_layer.dilation,
            bias=False).to(device)

        self.core_layer = nn.Conv2d(
            in_channels=in_rank,
            out_channels=out_rank,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            bias=False).to(device)

        self.out_channel_layer = nn.Conv2d(
            in_channels=out_rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=conv_layer.dilation,
            bias=conv_layer.bias is not None).to(device)

        if hooi:
            # use traditional tucker2 decomposition

            core, factors = decomp.partial_tucker(
                weight, modes=[0, 1], ranks=(out_rank, in_rank), init='svd')
            (out_channel_factor, in_channel_factor) = factors

            # assign bias
            if self.out_channel_layer.bias is not None:
                self.out_channel_layer.bias.data = conv_layer.bias.data

            # assign weights
            transposed = torch.transpose(in_channel_factor, 1, 0)
            self.in_channel_layer.weight.data = \
                transposed.unsqueeze(-1).unsqueeze(-1)
            self.out_channel_layer.weight.data = \
                out_channel_factor.unsqueeze(-1).unsqueeze(-1)
            self.core_layer.weight.data = core

    def forward(self, x):
        """
        Run forward propagation
        """
        x = self.in_channel_layer(x)
        x = self.core_layer(x)
        x = self.out_channel_layer(x)

        return x

    def recover(self):
        """
        Recover original tensor from decomposed tensor

        @return: 4D weight tensor with original layer's shape
        """
        core = self.core_layer.weight.data
        out_factor = self.out_channel_layer.weight.data.squeeze()
        in_factor = self.in_channel_layer.weight.data.squeeze()
        in_factor = torch.transpose(in_factor, 1, 0)

        recovered = tucker_to_tensor(core, [out_factor, in_factor])
        return recovered


class DecomposedLinear(Module):
    def __init__(self, layer: Module, ranks: tuple, hooi: bool = False):
        super(DecomposedLinear, self).__init__()

        device = layer.weight.device
        weight = layer.weight.data
        out_dim, in_dim = weight.shape
        out_rank, in_rank = ranks

        self.in_layer = nn.Linear(
            in_features=in_dim,
            out_features=in_rank,
            bias=False).to(device)

        self.core_layer = nn.Linear(
            in_features=in_rank,
            out_features=out_rank,
            bias=False).to(device)

        self.out_layer = nn.Linear(
            in_features=out_rank,
            out_features=out_dim,
            bias=layer.bias is not None).to(device)

        if hooi:
            core, factors = decomp.tucker(weight, ranks=ranks, init='svd')
            out_factor, in_factor = factors

            if self.out_layer.bias is not None:
                self.out_layer.bias.data = layer.bias.data

            self.in_layer.weight.data = torch.transpose(in_factor, 1, 0)
            self.out_layer.weight.data = out_factor
            self.core_layer.weight.data = core

    def forward(self, x):
        x = self.in_layer(x)
        x = self.core_layer(x)
        x = self.out_layer(x)
        return x

    def recover(self):
        pass
        # core = self.core_layer.weight.data
        # out_factor = self.out_channel_layer.weight.data.squeeze()
        # in_factor = self.in_channel_layer.weight.data.squeeze()
        # in_factor = torch.transpose(in_factor, 1, 0)
        #
        # recovered = tucker_to_tensor(core, [out_factor, in_factor])
        # return recovered
