import torch
import random
import numpy as np
import torch.nn as nn
from typing import Optional

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def z_score(x: torch.Tensor, mean: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
    """
    Z-score标准化
    """
    if mean is None:
        mean = x.mean(dim=-1, keepdim=True)
    if std is None:
        std = x.std(dim=-1, keepdim=True)
    return (x - mean) / std # (batch_size, seq_length, hidden_dim)

def inv_z_score(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Z-score反标准化
    """
    return x * std + mean # (batch_size, seq_length, hidden_dim)