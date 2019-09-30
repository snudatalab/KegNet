import abc
import os

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
from numpy import ndarray
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

ROOT_PATH = '../data'


def _normalize(arr: ndarray) -> ndarray:
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    arr = arr - avg
    arr[:, std != 0] /= std[std != 0]
    return arr


def _split2(nd: int, seed: int = 2019):
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    size = int(nd * 7 / 8)
    index1 = shuffled_index[:size]
    index2 = shuffled_index[size:]
    return index1, index2


def _split3(nd: int, seed: int = 2019):
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    index1 = shuffled_index[:int(nd * 0.7)]
    index2 = shuffled_index[int(nd * 0.7):int(nd * 0.8)]
    index3 = shuffled_index[int(nd * 0.8):]
    return index1, index2, index3


def _get_samplers(num_data: int, num_valid_data: int, seed: int):
    indices = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:-num_valid_data])
    valid_sampler = SubsetRandomSampler(indices[-num_valid_data:])
    return train_sampler, valid_sampler


def _to_image_loaders(trn_data, val_data, test_data, batch_size: int):
    samplers = _get_samplers(len(trn_data), 5000, seed=2019)
    trn_l = DataLoader(trn_data, batch_size, sampler=samplers[0])
    val_l = DataLoader(val_data, batch_size, sampler=samplers[1])
    test_l = DataLoader(test_data, batch_size)
    return trn_l, val_l, test_l


class Data:
    def __init__(self):
        self.nx = None
        self.ny = None
        self.nc = None
        self.size = None

    @abc.abstractmethod
    def to_loaders(self, batch_size: int) -> tuple:
        pass


class MNIST(Data):
    def __init__(self):
        super().__init__()
        self.nx = 1024
        self.ny = 10
        self.nc = 1
        self.size = 1, 32, 32

    def to_loaders(self, batch_size: int):
        path = '{}/mnist'.format(ROOT_PATH)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trn_data = torch_datasets.MNIST(
            path, train=True, transform=transform, download=True)
        test_data = torch_datasets.MNIST(path, train=False, transform=transform)
        return _to_image_loaders(trn_data, trn_data, test_data, batch_size)


class Fashion(Data):
    def __init__(self):
        super().__init__()
        self.nx = 1024
        self.ny = 10
        self.nc = 1
        self.size = 1, 32, 32

    def to_loaders(self, batch_size: int):
        path = '{}/fashion'.format(ROOT_PATH)
        stat = ((0.2856,), (0.3385,))

        train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])
        test_trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*stat)])

        trn_data = torch_datasets.FashionMNIST(
            path, train=True, transform=train_trans, download=True)
        val_data = torch_datasets.FashionMNIST(path, train=True, transform=test_trans)
        test_data = torch_datasets.FashionMNIST(path, train=False, transform=test_trans)
        return _to_image_loaders(trn_data, val_data, test_data, batch_size)


class SVHN(Data):
    def __init__(self):
        super().__init__()
        self.nx = 1024 * 3
        self.ny = 10
        self.nc = 3
        self.size = 3, 32, 32

    def to_loaders(self, batch_size: int):
        path = '{}/svhn'.format(ROOT_PATH)
        stat = ((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stat)])

        trn_data = torch_datasets.SVHN(path, 'train', trans, download=True)
        test_data = torch_datasets.SVHN(path, 'test', trans, download=True)
        return _to_image_loaders(trn_data, trn_data, test_data, batch_size)


class UCI(Data):
    @staticmethod
    def _read_dfs(dataset: str, path: str):
        df_list = []
        for mode in ['-', 'train', 'test']:
            if mode == '-':
                filename = '{}_R.dat'.format(dataset)
            else:
                filename = '{}_{}_R.dat'.format(dataset, mode)
            file = os.path.join(path, 'uci', dataset, filename)

            if os.path.exists(file):
                df = pd.read_csv(file, sep='\t', index_col=0)
                df_list.append(df.reset_index(drop=True))
        return df_list

    @staticmethod
    def _preprocess(df_list: list):
        if len(df_list) == 1:
            df = df_list[0]
            arr_x = df.iloc[:, :-1].values.astype(np.float32)
            arr_y = df.iloc[:, -1].values

            nd = arr_x.shape[0]
            nx = arr_x.shape[1]
            ny = arr_y.max() + 1

            train_index, valid_index, test_index = _split3(nd)

            trn_x = arr_x[train_index]
            trn_y = arr_y[train_index]
            val_x = arr_x[valid_index]
            val_y = arr_y[valid_index]
            test_x = arr_x[test_index]
            test_y = arr_y[test_index]

        elif len(df_list) == 2:
            trn_df = df_list[0]
            test_df = df_list[1]

            trn_x = trn_df.iloc[:, :-1].values.astype(np.float32)
            trn_y = trn_df.iloc[:, -1].values
            test_x = test_df.iloc[:, :-1].values.astype(np.float32)
            test_y = test_df.iloc[:, -1].values

            trn_x = _normalize(trn_x)
            test_x = _normalize(test_x)

            nd = trn_df.shape[0]
            nx = trn_x.shape[1]
            ny = trn_y.max() + 1
            trn_index, val_index = _split2(nd)

            val_x = trn_x[val_index, :]
            val_y = trn_y[val_index]
            trn_x = trn_x[trn_index, :]
            trn_y = trn_y[trn_index]

        else:
            raise ValueError()

        return trn_x, trn_y, val_x, val_y, test_x, test_y, nx, ny

    def __init__(self, dataset: str):
        super().__init__()
        df_list = self._read_dfs(dataset, path='../data')
        trn_x, trn_y, val_x, val_y, test_x, test_y, nx, ny = self._preprocess(df_list)
        self.nx = nx
        self.ny = ny
        self.nc = None
        self.size = self.nx,

        self.trn_data = torch.tensor(trn_x), torch.tensor(trn_y)
        self.val_data = torch.tensor(val_x), torch.tensor(val_y)
        self.test_data = torch.tensor(test_x), torch.tensor(test_y)

    def to_loaders(self, batch_size: int) -> tuple:
        trn_l = DataLoader(TensorDataset(*self.trn_data), batch_size)
        val_l = DataLoader(TensorDataset(*self.val_data), batch_size)
        test_l = DataLoader(TensorDataset(*self.test_data), batch_size)
        return trn_l, val_l, test_l


def to_dataset(dataset):
    """
    Return a dataset class given its name.
    """
    if dataset == 'mnist':
        return MNIST()
    elif dataset == 'fashion':
        return Fashion()
    elif dataset == 'svhn':
        return SVHN()
    else:
        return UCI(dataset)


def get_uci_datasets() -> list:
    return ['letter',
            'pendigits',
            'statlog-shuttle']


def is_uci(dataset: str) -> bool:
    return os.path.exists('../data/uci/{}'.format(dataset))
