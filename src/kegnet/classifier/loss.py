from torch import nn


class KLDivLoss(nn.Module):
    """
    Class for a KL divergence loss for knowledge distillation.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels):
        """
        Forward propagation.
        """
        return self.loss(self.log_softmax(outputs), labels)
