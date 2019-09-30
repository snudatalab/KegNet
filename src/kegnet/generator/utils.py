import numpy as np
import torch
from torch import Tensor

from kegnet.generator import models
from kegnet.utils import data


def generate_noises(size: tuple, dist: str):
    if dist == 'normal':
        return torch.randn(size)
    elif dist == 'binary':
        return torch.randint(0, 2, size, dtype=torch.float)
    elif dist == 'ternary':
        return torch.randint(-1, 2, size, dtype=torch.float)
    elif dist == 'onehot':
        values = np.random.randint(0, size[1], size[0])
        noises = torch.zeros(size)
        noises[torch.arange(size[0]), values] = 1
        return noises
    else:
        raise ValueError(dist)


def generate_labels(num_data: int, num_classes: int, dist: str) -> Tensor:
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


def init_generator(dataset: str) -> models.Generator:
    ny = data.to_dataset(dataset).ny
    nx = data.to_dataset(dataset).nx
    nc = data.to_dataset(dataset).nc
    nz = 10

    if dataset in ('mnist', 'fashion', 'svhn', 'cifar10'):
        return models.ImageGenerator(nz, ny, nc)
    elif data.is_uci(dataset):
        return models.DenseGenerator(nz, ny, nx, n_layers=2)
    else:
        raise ValueError(dataset)


def generate_data(generator: models.Generator,
                  repeats: int,
                  adjust: bool = True,
                  device: torch.device = None):
    generator.eval()

    num_noises = generator.num_noises
    num_classes = generator.num_classes

    noises = generate_noises((repeats, num_noises), dist='normal')
    noises[0, :] = 0
    noises = np.repeat(noises.detach().numpy(), repeats=num_classes, axis=0)
    noises = torch.tensor(noises, dtype=torch.float32, device=device)

    labels = np.zeros((num_classes, num_classes))
    # labels[:, 0] = np.array(range(9, -1, -1)) / 9
    # labels[:, 5] = np.array(range(0, 10)) / 9
    labels[np.arange(num_classes), np.arange(num_classes)] = 1
    labels = np.tile(labels, (repeats, 1))
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    fake_data = generator.forward(labels, noises, adjust)
    return fake_data.view(repeats, -1, *fake_data.shape[1:])
