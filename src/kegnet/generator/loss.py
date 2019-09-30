import torch
from torch import nn, Tensor
from torch.nn import Module


class ReconstructionLoss(Module):
    def __init__(self, how: str = 'kld'):
        super().__init__()
        self.how = how
        self.loss = self._to_loss(how)
        if how == 'kld':
            self.log_softmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def _to_loss(how: str) -> Module:
        if how == 'kld':
            return nn.KLDivLoss(reduction='batchmean')
        elif how == 'l1':
            return nn.L1Loss()
        elif how == 'l2':
            return nn.MSELoss()
        else:
            raise ValueError(how)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        if self.how == 'kld':
            output = self.log_softmax(output)
        return self.loss(output, target)


class DiversityLoss(nn.Module):
    def __init__(self, metric: str = 'l1'):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def _compute_distance(self, t1: Tensor, t2: Tensor, how: str):
        if how == 'l1':
            return torch.abs(t1 - t2).mean(dim=(2,))
        elif how == 'l2':
            return torch.pow(t1 - t2, 2).mean(dim=(2,))
        elif how == 'cosine':
            return 1 - self.cosine(t1, t2)
        else:
            raise ValueError(how)

    def _pairwise_distance(self, tensor: Tensor, how: str) -> Tensor:
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self._compute_distance(tensor1, tensor2, how)

    def forward(self, noises: Tensor, layer: Tensor) -> Tensor:
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self._pairwise_distance(layer, how=self.metric)
        noise_dist = self._pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
