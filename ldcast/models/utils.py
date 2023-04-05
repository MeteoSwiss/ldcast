import torch
from torch import nn


def normalization(channels, norm_type="group", num_groups=32):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif (not norm_type) or (norm_type.tolower() == 'none'):
        return nn.Identity()
    else:
        raise NotImplementedError(norm)


def activation(act_type="swish"):
    if act_type == "swish":
        return nn.SiLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif not act_type:
        return nn.Identity()
    else:
        raise NotImplementedError(act_type)
