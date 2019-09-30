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
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from kegnet.generator import models
from kegnet.utils import data
from kegnet.utils import utils


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


def sample_kegnet_data(dataset, num_data, generators, device):
    """
    Sample artificial data using generator networks.
    """
    gen_models = []
    for path in generators:
        generator = init_generator(dataset).to(device)
        utils.load_checkpoints(generator, path, device)
        generator.eval()
        gen_models.append(generator)

    ny = gen_models[0].num_classes
    nz = gen_models[0].num_noises
    noises = sample_noises(size=(num_data, nz))
    labels_in = sample_labels(num_data, ny, dist='onehot')
    loader = DataLoader(TensorDataset(noises, labels_in), batch_size=256)

    images_list = []
    for idx, generator in enumerate(gen_models):
        l1 = []
        for z, y in loader:
            z = z.to(device)
            y = y.to(device)
            l1.append(generator(y, z).detach())
        images_list.append(torch.cat(tuple(l1), dim=0))
    return torch.cat(tuple(images_list), dim=0)


def sample_random_data(dataset, num_data, dist, device):
    """
    Sample artificial data from simple distributions.
    """
    size = (num_data, *data.to_dataset(dataset).size)
    if dist == 'normal':
        return torch.randn(size, device=device)
    elif dist == 'uniform':
        tensor = torch.zeros(size, dtype=torch.float, device=device)
        tensor.uniform_(-1, 1)
        return tensor
    else:
        raise ValueError(dist)


def visualize_images(generator, path, device, repeats=10):
    """
    Generate and visualize data for a generator.
    """
    generator.eval()
    nz = generator.num_noises
    ny = generator.num_classes

    noises = sample_noises(size=(repeats, nz))
    noises[0, :] = 0
    noises = np.repeat(noises.detach().numpy(), repeats=ny, axis=0)
    noises = torch.tensor(noises, dtype=torch.float32, device=device)

    labels = np.zeros((ny, ny))
    labels[np.arange(ny), np.arange(ny)] = 1
    labels = np.tile(labels, (repeats, 1))
    labels = torch.tensor(labels, dtype=torch.float32, device=device)

    images = generator(labels, noises)
    images = images.view(repeats, -1, *images.shape[1:])
    images = images.view(-1, *images.shape[2:])

    save_image(images.detach(), path, nrow=repeats, normalize=True)
