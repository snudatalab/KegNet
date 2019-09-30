import os
import shutil

import numpy as np
import torch
import torch.optim as optim
from torch import optim as op
from torchvision.utils import save_image

from kegnet.classifier import utils as cls_utils
from kegnet.generator import loss as gen_loss
from kegnet.generator import models
from kegnet.generator import utils as gen_utils
from kegnet.utils import utils, data

DEVICE = None


def update(networks: tuple, losses: tuple, optimizer: optim.Adam, **kwargs) -> tuple:
    generator, classifier, decoder = networks
    cls_loss, dec_loss, div_loss = losses
    list_bs, list_loss, list_corr = [], [], []

    batch_size = kwargs['batch_size']
    num_batches = kwargs['num_batches']
    label_dist = kwargs['label_dist']
    noise_dist = kwargs['noise_dist']
    alpha = kwargs['alpha']
    beta = kwargs['beta']

    generator.train()

    n_classes = generator.num_classes
    n_noises = generator.num_noises

    for _ in range(num_batches):
        noise_size = batch_size, n_noises
        noises = gen_utils.generate_noises(noise_size, noise_dist).to(DEVICE)
        labels = gen_utils.generate_labels(batch_size, n_classes, label_dist).to(DEVICE)

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


def visualize_images(generator: models.Generator, epoch: int, path: str):
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    path_ = os.path.join(path, 'images/images-{:03d}.png'.format(epoch))
    images = gen_utils.generate_data(generator, repeats=10, device=DEVICE)
    images = images.view(-1, *images.shape[2:])
    save_image(images.detach(), path_, nrow=10, normalize=True)


def set_if_none(variable, value):
    return value if variable is None else variable


def main(dataset: str, classifier: str, index: int, path_out: str,
         model: str = None, alpha: float = None, beta: float = None) -> str:
    global DEVICE
    DEVICE = utils.set_device(gpu=index)
    utils.set_seed(seed=2019 + index)

    num_epochs = 200
    batch_size = 256
    num_batches = 100
    save_every = 100
    viz_every = 10
    label_dist = 'onehot'
    noise_dist = 'normal'

    if dataset == 'mnist':
        dec_layers = 1
        lrn_rate = 1e-3
        model = set_if_none(model, 'lenet5')
        alpha = set_if_none(alpha, 1)
        beta = set_if_none(beta, 0)
    elif dataset == 'fashion':
        dec_layers = 3
        lrn_rate = 1e-2
        model = set_if_none(model, 'resnet14')
        alpha = set_if_none(alpha, 1)
        beta = set_if_none(beta, 10)
    elif dataset == 'svhn':
        dec_layers = 3
        lrn_rate = 1e-2
        model = set_if_none(model, 'resnet14')
        alpha = set_if_none(alpha, 1)
        beta = set_if_none(beta, 1)
    elif data.is_uci(dataset):
        dec_layers = 2
        lrn_rate = 1e-4
        model = set_if_none(model, 'linear')
        alpha = set_if_none(alpha, 1)
        beta = set_if_none(beta, 0)
    else:
        raise ValueError()

    cls_network = cls_utils.init_classifier(dataset, model).to(DEVICE)
    gen_network = gen_utils.init_generator(dataset).to(DEVICE)
    utils.load_checkpoints(cls_network, classifier, DEVICE)

    num_noises = gen_network.num_noises
    num_features = data.to_dataset(dataset).nx
    dec_network = models.Decoder(num_features, num_noises, dec_layers).to(DEVICE)

    networks = gen_network, cls_network, dec_network

    path_loss = os.path.join(path_out, 'loss-gen.txt')
    path_model = os.path.join(path_out, 'generator')
    path_result = None

    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out)
    with open(path_loss, 'w') as f:
        f.write('Epoch\tClsLoss\tDecLoss\tDivLoss\tLossSum\tAccr\n')

    loss1 = gen_loss.ReconstructionLoss(how='kld').to(DEVICE)
    loss2 = gen_loss.ReconstructionLoss(how='l2').to(DEVICE)
    loss3 = gen_loss.DiversityLoss(metric='l1').to(DEVICE)
    losses = loss1, loss2, loss3

    params = list(gen_network.parameters()) + list(dec_network.parameters())
    optimizer = op.Adam(params, lrn_rate)

    for epoch in range(1, num_epochs + 1):
        trn_acc, trn_losses = update(
            networks, losses, optimizer,
            batch_size=batch_size,
            num_batches=num_batches,
            label_dist=label_dist,
            noise_dist=noise_dist,
            alpha=alpha,
            beta=beta)

        with open(path_loss, 'a') as f:
            f.write('{:3d}'.format(epoch))
            for loss in trn_losses:
                f.write('\t{:.8f}'.format(loss))
            f.write('\t{:.8f}\n'.format(trn_acc))

        if viz_every > 0 and epoch % viz_every == 0:
            visualize_images(gen_network, epoch, path_out)

        if epoch % save_every == 0:
            p = '{}-{:03d}.pth.tar'.format(path_model, epoch)
            utils.save_checkpoints(gen_network, p)
            path_result = p

    print('Finished training the generator (index={}).'.format(index))
    return path_result
