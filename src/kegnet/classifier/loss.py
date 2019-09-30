import torch
from torch import nn


class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        return self.loss(self.log_softmax(outputs), labels)
