from torch import nn

from kegnet.classifier.models import lenet, resnet, linear
from kegnet.utils import data


def init_classifier(dataset: str, model: str) -> nn.Module:
    d = data.to_dataset(dataset)
    if model == 'lenet5':
        return lenet.LeNet5()
    elif model == 'resnet14':
        return resnet.resnet14(num_classes=d.ny, num_channels=d.nc)
    elif model == 'resnet20':
        return resnet.resnet20(num_classes=d.ny, num_channels=d.nc)
    elif model == 'resnet56':
        return resnet.resnet56(num_classes=d.ny, num_channels=d.nc)
    elif model == 'linear':
        return linear.FNN(d.nx, d.ny)
    else:
        raise ValueError(dataset)


def compress_classifier(classifier: nn.Module, option: int):
    if isinstance(classifier, lenet.LeNet5):
        if option == 1:
            classifier.compress(target=(3,), rank='evbmf', hooi=True)
        elif option == 2:
            classifier.compress(target=(2, 3), rank='evbmf', hooi=True)
        elif option == 3:
            classifier.compress(target=(2,), rank='evbmf', hooi=True)
            classifier.compress(target=(3,), rank=(5, 8), hooi=True)
        else:
            raise ValueError()
    elif isinstance(classifier, resnet.ResNet):
        if option == 1:
            classifier.compress(target=(1, 2), rank='evbmf', hooi=True)
        elif option == 2:
            classifier.compress(target=(1,), rank='evbmf', hooi=True)
        elif option == 3:
            classifier.compress(target=(2,), rank='evbmf', hooi=True)
        else:
            raise ValueError()
    elif isinstance(classifier, linear.FNN):
        if option == 1:
            classifier.compress(units=10, hooi=True)
        elif option == 2:
            classifier.compress(units=5, hooi=True)
        else:
            raise ValueError()
    else:
        raise ValueError()
