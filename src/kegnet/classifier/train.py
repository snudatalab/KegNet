import os
from typing import Callable

import numpy as np
import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from kegnet.classifier import loss as cls_loss
from kegnet.generator import utils as gen_utils
from kegnet.classifier import utils as cls_utils
from kegnet.utils import data, utils

DEVICE = None
L_RATE = 1e-2
SAVE_EVERY = -1
VAL_EPOCHS = 10000
MIN_EPOCHS = 10000
MAX_EPOCHS = 0

DATASET = None
BATCH_SIZE = 64
NUM_BATCHES = 100
TEMPERATURE = 1


def set_model(dataset: str, model: str) -> str:
    if model is None:
        if dataset == 'mnist':
            return 'lenet5'
        elif dataset in ('fashion', 'svhn'):
            return 'resnet14'
        elif data.is_uci(dataset):
            return 'linear'
    else:
        return model


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


def prepare_data(data_dist: str,
                 teacher: Module,
                 loader: DataLoader = None,
                 generators: list = None):
    if data_dist == 'real':
        loader_out = loader
        loss_out = nn.CrossEntropyLoss().to(DEVICE)
    elif data_dist == 'kegnet':
        loader_out = utils.prepare_fake_data(
            teacher,
            generators=generators,
            device=DEVICE,
            dataset=DATASET,
            batch_size=BATCH_SIZE,
            num_batches=NUM_BATCHES,
            temperature=TEMPERATURE,
            adjust=True)
        loss_out = cls_loss.KLDivLoss().to(DEVICE)
    elif data_dist in ('uniform', 'normal'):
        loader_out = utils.prepare_random_data(
            teacher,
            dataset=DATASET,
            batch_size=BATCH_SIZE,
            num_batches=NUM_BATCHES,
            dist=data_dist,
            temperature=TEMPERATURE,
            device=DEVICE)
        loss_out = cls_loss.KLDivLoss().to(DEVICE)
    else:
        raise ValueError()

    return loader_out, loss_out


def compress_classifier(model: Module, option: int, path: str):
    size_before = utils.count_parameters(model)
    cls_utils.compress_classifier(model, option)
    size_after = utils.count_parameters(model)
    with open(path, 'w') as f:
        f.write('Parameters (before compression): {}\n'.format(size_before))
        f.write('Parameters (after compression): {}\n'.format(size_after))
        f.write('Compression ratio: {:2f}\n'.format(size_before / size_after))


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
         model: str = None,
         teacher: str = None,
         generators: list = None,
         option: int = None):
    global DEVICE, DATASET
    DATASET = dataset
    DEVICE = utils.set_device(gpu=index)
    set_parameters(dataset, data_dist)
    index = 0 if index is None else index
    utils.set_seed(seed=2019 + index)

    model = set_model(dataset, model)

    path_loss = os.path.join(path_out, get_loss_name(data_dist, option))
    path_model = os.path.join(path_out, 'classifier')
    path_comp = os.path.join(path_out, 'compression-{}.txt'.format(option))
    os.makedirs(path_out, exist_ok=True)

    model_t = cls_utils.init_classifier(DATASET, model).to(DEVICE)
    if teacher is not None:
        utils.load_checkpoints(model_t, teacher, DEVICE)

    loaders = data.to_dataset(DATASET).to_loaders(BATCH_SIZE)
    test_loss = nn.CrossEntropyLoss().to(DEVICE)
    train_data = prepare_data(data_dist, model_t, loaders[0], generators)
    valid_data = loaders[1], test_loss
    test_data = loaders[2], test_loss

    if data_dist != 'real':
        compress_classifier(model_t, option, path_comp)
    model_s = model_t

    optimizer = optim.Adam(model_s.parameters(), L_RATE)

    if train:
        initialize_loss_file(path_loss)

        best_acc, best_epoch = 0, 0
        for epoch in range(MAX_EPOCHS + 1):
            if epoch > 0:
                learn_classifier(model_s, *train_data, optimizer=optimizer)
            train_results = eval_classifier(model_s, *train_data)
            valid_results = eval_classifier(model_s, *valid_data)
            test_results = eval_classifier(model_s, *test_data)

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
