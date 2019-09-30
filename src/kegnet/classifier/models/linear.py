import torch
from torch import nn

from kegnet.utils import tucker


class MLP(nn.Module):
    def _to_layers(self, units: int):
        in_features = self.in_features
        drop_probability = self.drop_prob
        n_layers = self.n_layers
        num_classes = self.num_classes

        layers = [nn.Linear(in_features, units),
                  nn.ELU(),
                  nn.Dropout(drop_probability)]

        for n in range(n_layers):
            layers.extend([nn.Linear(units, units),
                           nn.ELU(),
                           nn.Dropout(drop_probability)])

        layers.append(nn.Linear(units, num_classes))
        return nn.Sequential(*layers)

    def __init__(self, in_features: int, num_classes: int, n_layers: int = 10):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.units = 100
        self.drop_prob = 0.15
        self.n_layers = n_layers
        self.layers = self._to_layers(self.units)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def compress(self, units: int, hooi: bool):
        layers = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                new_ranks = []
                for rank in layer.weight.shape:
                    if rank == self.units:
                        new_ranks.append(int(units))
                    else:
                        new_ranks.append(rank)
                layer = tucker.DecomposedLinear(layer, tuple(new_ranks), hooi=hooi)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)
