import os
from typing import Callable

import numpy as np
import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from kegnet.classifier import loss as cls_loss
from kegnet.classifier import utils as cls_utils
from kegnet.utils import data, utils
from kegnet.generator import utils as gen_utils

DEVICE = None
L_RATE = 1e-2
SAVE_EVERY = -1
VAL_EPOCHS = 10000
MIN_EPOCHS = 10000
MAX_EPOCHS = 0

DATASET = None
BATCH_SIZE = 64
NUM_BATCHES = 100


def set_parameters_for_teacher(dataset: str):
    global SAVE_EVERY, L_RATE, MIN_EPOCHS, VAL_EPOCHS, MAX_EPOCHS

    if dataset == 'mnist':
        L_RATE = 1e-5
        SAVE_EVERY = 10
        MAX_EPOCHS = 200
    elif dataset == 'fashion':
        L_RATE = 1e-2
        SAVE_EVERY = 20
        MAX_EPOCHS = 100
    elif dataset == 'svhn':
        L_RATE = 1e-4
        SAVE_EVERY = 10
        MAX_EPOCHS = 50
    elif data.is_uci(dataset):
        L_RATE = 1e-3
        SAVE_EVERY = 0
        VAL_EPOCHS = 20
        MIN_EPOCHS = 100
        MAX_EPOCHS = 1000
    else:
        raise ValueError()


def set_parameters_for_kegnet(dataset: str):
    global SAVE_EVERY, L_RATE, MIN_EPOCHS, VAL_EPOCHS, MAX_EPOCHS

    SAVE_EVERY = -1
    VAL_EPOCHS = 50
    MIN_EPOCHS = 100
    MAX_EPOCHS = 400

    if dataset == 'mnist':
        L_RATE = 1e-5
    elif dataset == 'fashion':
        L_RATE = 1e-4
    elif dataset == 'svhn':
        L_RATE = 1e-4
    elif data.is_uci(dataset):
        L_RATE = 1e-4
    else:
        raise ValueError()


def set_parameters_for_baselines(dataset: str):
    global SAVE_EVERY, L_RATE, MIN_EPOCHS, VAL_EPOCHS, MAX_EPOCHS

    SAVE_EVERY = -1
    VAL_EPOCHS = 50
    MIN_EPOCHS = 100
    MAX_EPOCHS = 1000

    if dataset == 'mnist':
        L_RATE = 1e-6
    elif dataset == 'fashion':
        L_RATE = 1e-4
    elif dataset == 'svhn':
        L_RATE = 1e-4
    elif data.is_uci(dataset):
        L_RATE = 1e-4
    else:
        raise ValueError()


def set_parameters(dataset: str, data_dist: str):
    if data_dist == 'real':
        set_parameters_for_teacher(dataset)
    elif data_dist == 'kegnet':
        set_parameters_for_kegnet(dataset)
    else:
        set_parameters_for_baselines(dataset)


def learn_classifier(classifier: Module, loader: DataLoader, loss_func: Callable,
                     optimizer: Adam):
    classifier.train()

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = classifier(images.to(DEVICE))
        loss = loss_func(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()


def eval_classifier(classifier: Module, loader: DataLoader, loss_func: Callable):
    classifier.eval()

    list_bs, list_loss, list_corr = [], [], []
    for data_x, data_y in loader:
        data_x = data_x.to(DEVICE)
        data_y = data_y.to(DEVICE)
        outputs = classifier(data_x)
        loss = loss_func(outputs, data_y)
        predicts = outputs.argmax(dim=1)
        if len(data_y.size()) > 1:
            data_y = data_y.argmax(dim=1)
        corrects = torch.sum(predicts.eq(data_y))

        list_bs.append(data_x.shape[0])
        list_loss.append(loss.item())
        list_corr.append(corrects.item())

    loss = np.average(list_loss, weights=list_bs)
    accuracy = np.sum(list_corr) / np.sum(list_bs)
    return loss, accuracy


def predict_labels(teacher, sampled_data, batch_size):
    teacher.eval()
    softmax = nn.Softmax(dim=1)
    loader = DataLoader(TensorDataset(sampled_data), batch_size=256)
    labels = []
    for x, in loader:
        labels.append(teacher(x).detach())
    labels = softmax(torch.cat(tuple(labels), dim=0))
    return DataLoader(TensorDataset(sampled_data, labels), batch_size)


def prepare_data(teacher, data_dist, generators=None):
    num_data = BATCH_SIZE * NUM_BATCHES
    if data_dist == 'kegnet':
        sampled_data = gen_utils.sample_kegnet_data(
            DATASET, num_data, generators, DEVICE)
    elif data_dist in ('uniform', 'normal'):
        sampled_data = gen_utils.sample_random_data(
            DATASET, num_data, data_dist, DEVICE)
    else:
        raise ValueError()

    return predict_labels(teacher, sampled_data, BATCH_SIZE)


def compress_classifier(model, option, path):
    size_before = cls_utils.count_parameters(model)
    model.compress(option)
    size_after = cls_utils.count_parameters(model)
    with open(path, 'w') as f:
        f.write(f'Parameters (before compression): {size_before}\n')
        f.write(f'Parameters (after compression): {size_after}\n')
        f.write(f'Compression ratio: {size_before / size_after:2f}\n')


def get_loss_name(data_dist: str, option: int) -> str:
    if option is None:
        return 'loss-{}.txt'.format(data_dist)
    else:
        return 'loss-{}-{}.txt'.format(data_dist, option)


def initialize_loss_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        f.write('Epoch\tTrnLoss\tTrnAccr\tValLoss\tValAccr\t'
                'TstLoss\tTstAccr\tIsBest\n')


def main(dataset: str,
         data_dist: str,
         index: int,
         path_out: str,
         train: bool = False,
         teacher: str = None,
         generators: list = None,
         option: int = None):
    global DEVICE, DATASET
    DATASET = dataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_parameters(dataset, data_dist)
    index = 0 if index is None else index
    utils.set_seed(seed=2019 + index)

    path_loss = os.path.join(path_out, get_loss_name(data_dist, option))
    path_model = os.path.join(path_out, 'classifier')
    path_comp = os.path.join(path_out, 'compression-{}.txt'.format(option))
    os.makedirs(path_out, exist_ok=True)

    model_t = cls_utils.init_classifier(DATASET).to(DEVICE)
    if teacher is not None:
        utils.load_checkpoints(model_t, teacher, DEVICE)

    loaders = data.to_dataset(DATASET).to_loaders(BATCH_SIZE)

    if data_dist == 'real':
        trn_loss = nn.CrossEntropyLoss().to(DEVICE)
    else:
        trn_loss = cls_loss.KLDivLoss().to(DEVICE)
    test_loss = nn.CrossEntropyLoss().to(DEVICE)

    if data_dist == 'real':
        trn_data = loaders[0]
    else:
        trn_data = prepare_data(model_t, data_dist, generators)
    val_data = loaders[1]
    test_data = loaders[2]

    if data_dist != 'real':
        compress_classifier(model_t, option, path_comp)
    model_s = model_t

    optimizer = optim.Adam(model_s.parameters(), L_RATE)

    if train:
        initialize_loss_file(path_loss)

        best_acc, best_epoch = 0, 0
        for epoch in range(MAX_EPOCHS + 1):
            if epoch > 0:
                learn_classifier(model_s, trn_data, trn_loss, optimizer)
            train_results = eval_classifier(model_s, trn_data, trn_loss)
            valid_results = eval_classifier(model_s, val_data, test_loss)
            test_results = eval_classifier(model_s, test_data, test_loss)

            _, valid_acc = valid_results
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = epoch

            if epoch > max(best_epoch + VAL_EPOCHS, MIN_EPOCHS):
                break

            if epoch > 0:
                if epoch == best_epoch:
                    p = '{}-best.pth.tar'.format(path_model)
                    utils.save_checkpoints(model_s, p)
                if SAVE_EVERY > 0 and epoch % SAVE_EVERY == 0:
                    p = '{}-{:03d}.pth.tar'.format(path_model, epoch)
                    utils.save_checkpoints(model_s, p)

            with open(path_loss, 'a') as f:
                f.write('{:3d}'.format(epoch))
                f.write('\t{:.8f}\t{:.8f}'.format(*train_results))
                f.write('\t{:.8f}\t{:.8f}'.format(*valid_results))
                f.write('\t{:.8f}\t{:.8f}'.format(*test_results))
                if epoch == best_epoch:
                    f.write('\tBEST')
                f.write('\n')

        print('Finished training the classifier (index={}).'.format(index))
    else:
        test_loss, test_acc = eval_classifier(model_s, *test_data)
        print('{}: loss = {}, accuracy = {}'.format(dataset, test_loss, test_acc))
