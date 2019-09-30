import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kegnet.generator import utils as gen_utils
from kegnet.utils import data


def set_device(gpu: int, n_gpu: int = 7):
    if gpu is None:
        return torch.device('cpu')
    else:
        gpu = gpu % n_gpu
        device = 'cuda:{}'.format(gpu) if gpu is not None else 'cuda'
        return torch.device(device if torch.cuda.is_available() else 'cpu')


def count_parameters(model: nn.Module) -> int:
    size = 0
    for parameter in model.parameters():
        size += np.prod(parameter.shape)
    return size


def prepare_fake_data(teacher: nn.Module, **kwargs) -> DataLoader:
    gen_paths = kwargs['generators']
    device = kwargs['device']
    dataset = kwargs['dataset']
    batch_size = kwargs['batch_size']
    num_batches = kwargs['num_batches']
    label_dist = kwargs['label_dist']
    adjust = kwargs['adjust']
    temperature = kwargs['temperature']

    teacher.eval()

    generators = []
    for path in gen_paths:
        generator = gen_utils.init_generator(dataset).to(device)
        load_checkpoints(generator, path, device)
        generator.eval()
        generators.append(generator)

    num_data = batch_size * num_batches
    num_classes = generators[0].num_classes
    num_noises = generators[0].num_noises

    noises = gen_utils.generate_noises(size=(num_data, num_noises), dist='normal')
    labels_in = gen_utils.generate_labels(num_data, num_classes, label_dist)
    loader = DataLoader(TensorDataset(noises, labels_in), batch_size=256)

    softmax = nn.Softmax(dim=1)
    images_list, labels_list = [], []
    for idx, generator in enumerate(generators):
        l1, l2 = [], []
        for z, s in loader:
            z, s = z.to(device), s.to(device)
            images = generator(s, z, adjust=adjust).detach()
            labels = softmax(teacher(images) / temperature).detach()
            l1.append(images)
            l2.append(labels)
        images_list.append(torch.cat(tuple(l1), dim=0))
        labels_list.append(torch.cat(tuple(l2), dim=0))

    dataset = TensorDataset(
        torch.cat(tuple(images_list), dim=0),
        torch.cat(tuple(labels_list)))
    return DataLoader(dataset, batch_size, shuffle=True)


def prepare_random_data(teacher: nn.Module, **kwargs) -> DataLoader:
    dataset = kwargs['dataset']
    batch_size = kwargs['batch_size']
    num_batches = kwargs['num_batches']
    dist = kwargs['dist']
    device = kwargs['device']
    temperature = kwargs['temperature']
    softmax = torch.nn.Softmax(dim=1)

    def to_images(size_: tuple):
        if dist == 'normal':
            return torch.randn(size_, device=device)
        elif dist == 'uniform':
            tensor = torch.zeros(size_, dtype=torch.float, device=device)
            tensor.uniform_(-1, 1)
            return tensor

    def to_labels(images_: torch.Tensor):
        loader = DataLoader(TensorDataset(images_), batch_size=batch_size)
        label_list = []
        for x, in loader:
            label_list.append(teacher(x).detach() / temperature)
        return softmax(torch.cat(tuple(label_list), dim=0))

    teacher.eval()
    num_data = batch_size * num_batches
    images = to_images(size_=(num_data, *data.to_dataset(dataset).size))
    labels = to_labels(images)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def set_gpu(gpu: int):
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def save_checkpoints(model: nn.Module or nn.DataParallel, path: str):
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model_state), path)


def load_checkpoints(model: nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get('model_state', None)
    model.load_state_dict(model_state)
