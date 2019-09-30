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
import tensorly as tl
import torch
from tensorly import decomposition as decomp
from tensorly import tucker_tensor as tucker
from torch import nn

from kegnet.utils import vbmf

tl.set_backend('pytorch')


class DecomposedConv2d(nn.Module):
    """
    Decomposed (or compressed) convolutional layer.
    """

    @staticmethod
    def choose_ranks(weight, ranks):
        """
        Choose the target ranks.
        """
        out_channels, in_channels, _, _ = weight.shape
        if ranks == 'evbmf':
            unfold_0 = tl.base.unfold(weight, 0)
            unfold_1 = tl.base.unfold(weight, 1)
            _, diag_0, _, _ = vbmf.EVBMF(unfold_0)
            _, diag_1, _, _ = vbmf.EVBMF(unfold_1)
            out_rank = diag_0.shape[0]
            in_rank = diag_1.shape[1]
        elif isinstance(ranks, float):
            out_rank = int(out_channels * ranks)
            in_rank = int(in_channels * ranks)
        elif isinstance(ranks, tuple):
            in_rank, out_rank = ranks
        else:
            raise ValueError(ranks)
        return out_rank, in_rank

    def __init__(self, layer, ranks='evbmf', init=True):
        """
        Class initializer.
        """
        super(DecomposedConv2d, self).__init__()

        device = layer.weight.device
        weight = layer.weight.data
        out_channels, in_channels, _, _ = weight.shape
        out_rank, in_rank = self.choose_ranks(weight, ranks)

        self.in_channel_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=layer.dilation,
            bias=False).to(device)

        self.core_layer = nn.Conv2d(
            in_channels=in_rank,
            out_channels=out_rank,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            bias=False).to(device)

        self.out_channel_layer = nn.Conv2d(
            in_channels=out_rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=layer.dilation,
            bias=layer.bias is not None).to(device)

        if init:
            core, factors = decomp.partial_tucker(
                weight, modes=[0, 1], ranks=(out_rank, in_rank), init='svd')
            (out_channel_factor, in_channel_factor) = factors

            if self.out_channel_layer.bias is not None:
                self.out_channel_layer.bias.data = layer.bias.data

            transposed = torch.transpose(in_channel_factor, 1, 0)
            self.in_channel_layer.weight.data = \
                transposed.unsqueeze(-1).unsqueeze(-1)
            self.out_channel_layer.weight.data = \
                out_channel_factor.unsqueeze(-1).unsqueeze(-1)
            self.core_layer.weight.data = core

    def forward(self, x):
        """
        Forward propagation.
        """
        x = self.in_channel_layer(x)
        x = self.core_layer(x)
        x = self.out_channel_layer(x)
        return x

    def recover(self):
        """
        Recover the original shape.
        """
        core = self.core_layer.weight.data
        out_factor = self.out_channel_layer.weight.data.squeeze()
        in_factor = self.in_channel_layer.weight.data.squeeze()
        in_factor = torch.transpose(in_factor, 1, 0)
        return tucker.tucker_to_tensor(core, [out_factor, in_factor])


class DecomposedLinear(nn.Module):
    """
    Decomposed (or compressed) linear layer.
    """

    def __init__(self, layer, ranks, init=True):
        """
        Class initializer.
        """
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

        if init:
            core, factors = decomp.tucker(weight, ranks=ranks, init='svd')
            out_factor, in_factor = factors

            if self.out_layer.bias is not None:
                self.out_layer.bias.data = layer.bias.data

            self.in_layer.weight.data = torch.transpose(in_factor, 1, 0)
            self.out_layer.weight.data = out_factor
            self.core_layer.weight.data = core

    def forward(self, x):
        """
        Forward propagation.
        """
        x = self.in_layer(x)
        x = self.core_layer(x)
        x = self.out_layer(x)
        return x
