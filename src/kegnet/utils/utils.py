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
import os

import numpy as np
import torch


def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_checkpoints(model, path):
    """
    Save a trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)


def load_checkpoints(model, path, device):
    """
    Load a saved model.
    """
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get('model_state', None)
    model.load_state_dict(model_state)
