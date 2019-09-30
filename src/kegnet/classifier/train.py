import os

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from kegnet.classifier import loss as cls_loss
from kegnet.classifier import utils as cls_utils
from kegnet.utils import data, utils
from kegnet.generator import utils as gen_utils

DEVICE = None


def update_classifier(classifier, loader, loss_func, optimizer):
    """
    Update a classifier network for a single epoch.
    """
    classifier.train()

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = classifier(images.to(DEVICE))
        loss = loss_func(outputs, labels.to(DEVICE))
        loss.backward()
        optimizer.step()


def eval_classifier(classifier, loader, loss_func):
    """
    Evaluate a classifier network.
    """
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


def predict_labels(model, sampled_data):
    """
    Predict the labels of sampled data.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    loader = DataLoader(TensorDataset(sampled_data), batch_size=256)
    labels = []
    for x, in loader:
        labels.append(model(x).detach())
    return softmax(torch.cat(tuple(labels), dim=0))


def prepare_data(model, data_dist, dataset, batch_size, num_batches,
                 generators=None):
    """
    Prepare a dataset to train a student network.
    """
    num_data = batch_size * num_batches
    if data_dist == 'kegnet':
        sampled_data = gen_utils.sample_kegnet_data(
            dataset, num_data, generators, DEVICE)
    elif data_dist in ('uniform', 'normal'):
        sampled_data = gen_utils.sample_random_data(
            dataset, num_data, data_dist, DEVICE)
    else:
        raise ValueError()
    labels = predict_labels(model, sampled_data)
    return DataLoader(TensorDataset(sampled_data, labels), batch_size)


def compress_classifier(model, option, path):
    """
    Compress a classifier based on the given option.
    """
    size_before = cls_utils.count_parameters(model)
    model.compress(option)
    size_after = cls_utils.count_parameters(model)
    with open(path, 'w') as f:
        f.write(f'Parameters (before compression): {size_before}\n')
        f.write(f'Parameters (after compression): {size_after}\n')
        f.write(f'Compression ratio: {size_before / size_after:2f}\n')


def prepare_teacher(dataset):
    """
    Prepare datasets and hyperparameters for training a teacher network.
    """
    batch_size = 64

    if dataset == 'mnist':
        lrn_rate = 1e-5
        save_every = 10
        min_epochs = 10000
        val_epochs = 10000
        max_epochs = 200
    elif dataset == 'fashion':
        lrn_rate = 1e-2
        save_every = 20
        min_epochs = 10000
        val_epochs = 10000
        max_epochs = 100
    elif dataset == 'svhn':
        lrn_rate = 1e-4
        save_every = 10
        min_epochs = 10000
        val_epochs = 10000
        max_epochs = 50
    else:
        lrn_rate = 1e-3
        save_every = 0
        val_epochs = 20
        min_epochs = 100
        max_epochs = 1000

    trn_data, val_data, test_data = data.to_dataset(dataset).to_loaders(batch_size)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)

    return dict(
        lrn_rate=lrn_rate,
        save_every=save_every,
        min_epochs=min_epochs,
        val_epochs=val_epochs,
        max_epochs=max_epochs,
        trn_data=trn_data,
        val_data=val_data,
        test_data=test_data,
        trn_loss_func=loss_func,
        test_loss_func=loss_func)


def prepare_student(model, dataset, data_dist, generators=None):
    """
    Prepare datasets and hyperparameters for training a student network.
    """
    batch_size = 64
    num_batches = 100
    save_every = -1
    val_epochs = 50
    min_epochs = 100

    if data_dist == 'kegnet':
        max_epochs = 400

        if dataset == 'mnist':
            lrn_rate = 1e-5
        else:
            lrn_rate = 1e-4
    elif data_dist in ('normal', 'uniform'):
        max_epochs = 1000

        if dataset == 'mnist':
            lrn_rate = 1e-6
        else:
            lrn_rate = 1e-4
    else:
        raise ValueError()

    trn_data = prepare_data(
        model, data_dist, dataset, batch_size, num_batches, generators)
    _, val_data, test_data = data.to_dataset(dataset).to_loaders(batch_size)
    trn_loss_func = cls_loss.KLDivLoss().to(DEVICE)
    test_loss_func = nn.CrossEntropyLoss().to(DEVICE)

    return dict(
        lrn_rate=lrn_rate,
        save_every=save_every,
        min_epochs=min_epochs,
        val_epochs=val_epochs,
        max_epochs=max_epochs,
        trn_data=trn_data,
        val_data=val_data,
        test_data=test_data,
        trn_loss_func=trn_loss_func,
        test_loss_func=test_loss_func)


def main(dataset, data_dist, path_out, index=0, load=None, generators=None,
         option=None):
    """
    Main function for training a classifier network.
    """
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(seed=2019 + index)

    path_loss = os.path.join(path_out, f'loss-{data_dist}-{option}.txt')
    path_model = os.path.join(path_out, 'classifier')
    path_comp = os.path.join(path_out, f'compression-{option}.txt')
    os.makedirs(path_out, exist_ok=True)

    model = cls_utils.init_classifier(dataset).to(DEVICE)
    if load is not None:
        utils.load_checkpoints(model, load, DEVICE)

    if data_dist == 'real':
        params = prepare_teacher(dataset)
    elif data_dist in ('kegnet', 'uniform', 'normal'):
        params = prepare_student(model, dataset, data_dist, generators)
        compress_classifier(model, option, path_comp)
    else:
        raise ValueError()

    lrn_rate = params['lrn_rate']
    save_every = params['save_every']
    min_epochs = params['min_epochs']
    val_epochs = params['val_epochs']
    max_epochs = params['max_epochs']
    trn_data = params['trn_data']
    val_data = params['val_data']
    test_data = params['test_data']
    trn_loss_func = params['trn_loss_func']
    test_loss_func = params['test_loss_func']

    optimizer = optim.Adam(model.parameters(), lrn_rate)

    with open(path_loss, 'w') as f:
        f.write('Epoch\tTrnLoss\tTrnAccuracy\tValLoss\tValAccuracy\t'
                'TestLoss\tTestAccuracy\tIsBest\n')

    best_acc, best_epoch = 0, 0
    for epoch in range(max_epochs + 1):
        if epoch > 0:
            update_classifier(model, trn_data, trn_loss_func, optimizer)
        trn_loss, trn_acc = eval_classifier(model, trn_data, trn_loss_func)
        val_loss, val_acc = eval_classifier(model, val_data, test_loss_func)
        test_loss, test_acc = eval_classifier(model, test_data, test_loss_func)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch

        if epoch > max(best_epoch + val_epochs, min_epochs):
            break

        if epoch > 0:
            if epoch == best_epoch:
                p = f'{path_model}-best.pth.tar'
                utils.save_checkpoints(model, p)
            if save_every > 0 and epoch % save_every == 0:
                p = f'{path_model}-{epoch:03d}.pth.tar'
                utils.save_checkpoints(model, p)

        with open(path_loss, 'a') as f:
            f.write(f'{epoch:3d}\t')
            f.write(f'{trn_loss:.8f}\t{trn_acc:.8f}\t')
            f.write(f'{val_loss:.8f}\t{val_acc:.8f}\t')
            f.write(f'{test_loss:.8f}\t{test_acc:.8f}')
            if epoch == best_epoch:
                f.write('\tBEST')
            f.write('\n')

    print(f'Finished training the classifier (index={index}).')
