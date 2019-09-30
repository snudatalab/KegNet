from torch import nn

from kegnet.utils import tucker


class MLP(nn.Module):
    """
    Class for a multilayer perceptron (MLP).
    """

    def __init__(self, in_features, num_classes, units=100, drop_prob=0.15,
                 n_layers=10):
        """
        Class initializer.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.units = units
        self.drop_prob = drop_prob
        self.n_layers = n_layers

        layers = []
        size_in = in_features
        for n in range(n_layers + 1):
            layers.extend([nn.Linear(size_in, units),
                           nn.ELU(),
                           nn.Dropout(drop_prob)])
            size_in = units
        layers.append(nn.Linear(units, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        return self.layers(x)

    def compress_layers(self, units):
        """
        Compress its layers.
        """
        layers = []
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                new_ranks = []
                for rank in layer.weight.shape:
                    if rank == self.units:
                        new_ranks.append(units)
                    else:
                        new_ranks.append(rank)
                layer = tucker.DecomposedLinear(layer, tuple(new_ranks))
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def compress(self, option):
        """
        Compress the network based on the option.
        """
        if option == 1:
            self.compress_layers(units=10)
        elif option == 2:
            self.compress_layers(units=5)
        else:
            raise ValueError()
