import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.dit import DiT
from models.diffusion.unet import UNet

def get_denoiser(**kwargs) -> nn.Module:
    """获取去噪器"""
    model_type = kwargs["type"].lower()
    if model_type == "dit":
        return DiT(**kwargs)
    elif model_type == "unet":
        return UNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")