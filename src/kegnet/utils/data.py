import os

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

ROOT_PATH = '../data'


def _normalize(arr):
    """
    Normalize a numpy array into zero-mean and unit-variance.
    """
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    arr = arr - avg
    arr[:, std != 0] /= std[std != 0]
    return arr


def _split2(nd, seed=2019):
    """
    Split data into the 7:1 ratios.
    """
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    size = int(nd * 7 / 8)
    index1 = shuffled_index[:size]
    index2 = shuffled_index[size:]
    return index1, index2


def _split3(nd, seed=2019):
    """
    Split data into the 7:1:2 ratios.
    """
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    index1 = shuffled_index[:int(nd * 0.7)]
    index2 = shuffled_index[int(nd * 0.7):int(nd * 0.8)]
    index3 = shuffled_index[int(nd * 0.8):]
    return index1, index2, index3


def _get_samplers(num_data, num_valid_data, seed):
    """
    Return a pair of samplers for the image datasets.
    """
    indices = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:-num_valid_data])
    valid_sampler = SubsetRandomSampler(indices[-num_valid_data:])
    return train_sampler, valid_sampler


def _to_image_loaders(trn_data, val_data, test_data, batch_size):
    """
    Convert an image dataset into loaders.
    """
    samplers = _get_samplers(len(trn_data), 5000, seed=2019)
    trn_l = DataLoader(trn_data, batch_size, sampler=samplers[0])
    val_l = DataLoader(val_data, batch_size, sampler=samplers[1])
    test_l = DataLoader(test_data, batch_size)
    return trn_l, val_l, test_l


class MNIST:
    """
    Class for the MNIST dataset.
    """
    nx = 1024
    ny = 10
    nc = 1
    size = 1, 32, 32

    @staticmethod
    def to_loaders(batch_size):
        """
        Convert the dataset into data loaders.
        """
        path = f'{ROOT_PATH}/mnist'
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        trn_data = torch_datasets.MNIST(
            path, train=True, transform=transform, download=True)
        test_data = torch_datasets.MNIST(
            path, train=False, transform=transform)
        return _to_image_loaders(trn_data, trn_data, test_data, batch_size)


class Fashion:
    """
    Class for the Fashion MNIST dataset.
    """
    nx = 1024
    ny = 10
    nc = 1
    size = 1, 32, 32

    @staticmethod
    def to_loaders(batch_size):
        """
        Convert the dataset into data loaders.
        """
        path = f'{ROOT_PATH}/fashion'
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
        val_data = torch_datasets.FashionMNIST(
            path, train=True, transform=test_trans)
        test_data = torch_datasets.FashionMNIST(
            path, train=False, transform=test_trans)
        return _to_image_loaders(trn_data, val_data, test_data, batch_size)


class SVHN:
    """
    Class for the SVHN dataset.
    """
    nx = 1024 * 3
    ny = 10
    nc = 3
    size = 3, 32, 32

    @staticmethod
    def to_loaders(batch_size):
        """
        Convert the dataset into data loaders.
        """
        path = f'{ROOT_PATH}/svhn'
        stat = ((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stat)])

        trn_data = torch_datasets.SVHN(path, 'train', trans, download=True)
        test_data = torch_datasets.SVHN(path, 'test', trans, download=True)
        return _to_image_loaders(trn_data, trn_data, test_data, batch_size)


class UCI:
    """
    Class for the UCI datasets.
    """

    def __init__(self, dataset):
        """
        Class initializer.
        """
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

    @staticmethod
    def _read_dfs(dataset, path):
        """
        Read DataFrames of raw data.
        """
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
    def _preprocess(df_list):
        """
        Preprocess a dataset based on its properties.
        """
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

    def to_loaders(self, batch_size):
        """
        Convert the dataset into data loaders.
        """
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
