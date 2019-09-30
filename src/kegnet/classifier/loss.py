"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
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
