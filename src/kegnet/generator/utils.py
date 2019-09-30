import numpy as np
import torch

from kegnet.generator import models
from kegnet.utils import data


def sample_noises(size):
    """
    Sample noise vectors (z).
    """
    return torch.randn(size)


def sample_labels(num_data, num_classes, dist):
    """
    Sample label vectors (y).
    """
    if dist == 'onehot':
        init_labels = np.random.randint(0, num_classes, num_data)
        labels = np.zeros((num_data, num_classes), dtype=int)
        labels[np.arange(num_data), init_labels] = 1
        return torch.tensor(labels, dtype=torch.float32)
    elif dist == 'uniform':
        labels = np.random.uniform(size=(num_data, num_classes))
        return torch.tensor(labels, dtype=torch.float32)
    else:
        raise ValueError(dist)


def init_generator(dataset):
    """
    Initialize a generator based on the dataset.
    """
    d = data.to_dataset(dataset)
    if dataset in ('mnist', 'fashion', 'svhn'):
        return models.ImageGenerator(d.ny, d.nc)
    else:
        return models.DenseGenerator(d.ny, d.nx, n_layers=2)
