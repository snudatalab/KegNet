import os

import numpy as np
import torch
from torch import optim

from kegnet.classifier import utils as cls_utils
from kegnet.generator import loss as gen_loss
from kegnet.generator import models
from kegnet.generator import utils as gen_utils
from kegnet.utils import utils, data

DEVICE = None


def update(networks, losses, optimizer, alpha, beta):
    """
    Update generator and decoder networks for a single epoch.
    """
    batch_size = 256
    num_batches = 100

    generator, classifier, decoder = networks
    cls_loss, dec_loss, div_loss = losses
    list_bs, list_loss, list_corr = [], [], []

    generator.train()

    n_classes = generator.num_classes
    n_noises = generator.num_noises

    for _ in range(num_batches):
        noise_size = batch_size, n_noises
        noises = gen_utils.sample_noises(noise_size).to(DEVICE)
        labels = gen_utils.sample_labels(batch_size, n_classes, dist='onehot').to(DEVICE)

        optimizer.zero_grad()

        images = generator(labels, noises)
        outputs = classifier(images)

        loss1 = cls_loss(outputs, labels)
        loss2 = dec_loss(decoder(images), noises) * alpha
        loss3 = div_loss(noises, images) * beta
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        corrects = torch.sum(outputs.argmax(dim=1).eq(labels.argmax(dim=1)))
        list_bs.append(batch_size)
        list_loss.append((loss1.item(), loss2.item(), loss3.item(), loss.item()))
        list_corr.append(corrects.item())

    loss = np.average(list_loss, axis=0, weights=list_bs)
    accuracy = np.sum(list_corr) / np.sum(list_bs)
    return accuracy, loss


def main(dataset, index, cls_path, out_path):
    """
    Main function for training a generator.
    """
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seed(seed=2019 + index)

    num_epochs = 200
    save_every = 100
    viz_every = 10

    assert num_epochs >= save_every

    if dataset == 'mnist':
        dec_layers = 1
        lrn_rate = 1e-3
        alpha = 1
        beta = 0
    elif dataset == 'fashion':
        dec_layers = 3
        lrn_rate = 1e-2
        alpha = 1
        beta = 10
    elif dataset == 'svhn':
        dec_layers = 3
        lrn_rate = 1e-2
        alpha = 1
        beta = 1
    else:
        dec_layers = 2
        lrn_rate = 1e-4
        alpha = 1
        beta = 0

    cls_network = cls_utils.init_classifier(dataset).to(DEVICE)
    gen_network = gen_utils.init_generator(dataset).to(DEVICE)
    utils.load_checkpoints(cls_network, cls_path, DEVICE)

    nz = gen_network.num_noises
    nx = data.to_dataset(dataset).nx
    dec_network = models.Decoder(nx, nz, dec_layers).to(DEVICE)

    networks = (gen_network, cls_network, dec_network)

    path_loss = os.path.join(out_path, 'loss-gen.txt')
    dir_model = os.path.join(out_path, 'generator')
    path_model = None

    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    with open(path_loss, 'w') as f:
        f.write('Epoch\tClsLoss\tDecLoss\tDivLoss\tLossSum\tAccuracy\n')

    loss1 = gen_loss.ReconstructionLoss(method='kld').to(DEVICE)
    loss2 = gen_loss.ReconstructionLoss(method='l2').to(DEVICE)
    loss3 = gen_loss.DiversityLoss(metric='l1').to(DEVICE)
    losses = loss1, loss2, loss3

    params = list(gen_network.parameters()) + list(dec_network.parameters())
    optimizer = optim.Adam(params, lrn_rate)

    for epoch in range(1, num_epochs + 1):
        trn_acc, trn_losses = update(
            networks, losses, optimizer, alpha, beta)

        with open(path_loss, 'a') as f:
            f.write(f'{epoch:3d}')
            for loss in trn_losses:
                f.write(f'\t{loss:.8f}')
            f.write(f'\t{trn_acc:.8f}\n')

        if viz_every > 0 and epoch % viz_every == 0:
            path = os.path.join(out_path, f'images/images-{epoch:03d}.png')
            gen_utils.visualize_images(gen_network, path, DEVICE)

        if epoch % save_every == 0:
            path = f'{dir_model}-{epoch:03d}.pth.tar'
            utils.save_checkpoints(gen_network, path)
            path_model = path

    print(f'Finished training the generator (index={index}).')
    return path_model
