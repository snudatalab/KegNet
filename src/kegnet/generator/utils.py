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


def generate_data(generator,
                  repeats: int,
                  adjust: bool = True,
                  device: torch.device = None):
    generator.eval()

    num_noises = generator.num_noises
    num_classes = generator.num_classes

    noises = sample_noises(size=(repeats, num_noises))
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
