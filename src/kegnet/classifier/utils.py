from torch import nn

from kegnet.classifier.models import lenet, resnet, linear
from kegnet.utils import data


def init_classifier(dataset: str, model: str) -> nn.Module:
    d = data.to_dataset(dataset)
    if model == 'lenet5':
        return lenet.LeNet5()
    elif model == 'resnet14':
        return resnet.resnet14(num_classes=d.ny, num_channels=d.nc)
    elif model == 'linear':
        return linear.MLP(d.nx, d.ny)
    else:
        raise ValueError(dataset)


def compress_classifier(classifier, option):
    if isinstance(classifier, lenet.LeNet5):
        classifier.compress(option)
    elif isinstance(classifier, resnet.ResNet):
        if option == 1:
            classifier.compress(target=(1, 2), rank='evbmf', hooi=True)
        elif option == 2:
            classifier.compress(target=(1,), rank='evbmf', hooi=True)
        elif option == 3:
            classifier.compress(target=(2,), rank='evbmf', hooi=True)
        else:
            raise ValueError()
    elif isinstance(classifier, linear.MLP):
        if option == 1:
            classifier.compress(units=10, hooi=True)
        elif option == 2:
            classifier.compress(units=5, hooi=True)
        else:
            raise ValueError()
    else:
        raise ValueError()
