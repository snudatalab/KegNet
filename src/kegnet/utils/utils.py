import os

import numpy as np
import torch
from torch import nn


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_checkpoints(model: nn.Module or nn.DataParallel, path: str):
    if isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model_state), path)


def load_checkpoints(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model_state = checkpoint.get('model_state', None)
    model.load_state_dict(model_state)
