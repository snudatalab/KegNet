from torch import nn

from kegnet.classifier.models import lenet, resnet, linear
from kegnet.utils import data


def init_classifier(dataset: str, model: str) -> nn.Module:
    d = data.to_dataset(dataset)
    if model == 'lenet5':
        return lenet.LeNet5()
    elif model == 'resnet14':
        return resnet.ResNet(num_classes=d.ny, num_channels=d.nc)
    elif model == 'linear':
        return linear.MLP(d.nx, d.ny)
    else:
        raise ValueError(dataset)
