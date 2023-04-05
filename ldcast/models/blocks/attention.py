import math

import torch
from torch import nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(
        self, channels, context_channels=None, 
        head_dim=32, num_heads=8
    ):
        super().__init__()
        self.channels = channels
        if context_channels is None:
            context_channels = channels
        self.context_channels = context_channels
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.attn_scale = self.head_dim ** -0.5
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.KV = nn.Linear(context_channels, self.inner_dim*2)
        self.Q = nn.Linear(channels, self.inner_dim)
        self.proj = nn.Linear(self.inner_dim, channels)

    def forward(self, x, y=None):
        if y is None:
            y = x
        
        (K,V) = self.KV(y).chunk(2, dim=-1)
        (B, Dk, H, W, C) = K.shape
        shape = (B, Dk, H, W, self.num_heads, self.head_dim)
        K = K.reshape(shape)
        V = V.reshape(shape)
        
        Q = self.Q(x)
        (B, Dq, H, W, C) = Q.shape
        shape = (B, Dq, H, W, self.num_heads, self.head_dim)
        Q = Q.reshape(shape)

        K = K.permute((0,2,3,4,5,1)) # K^T
        V = V.permute((0,2,3,4,1,5))
        Q = Q.permute((0,2,3,4,1,5))

        attn = torch.matmul(Q, K) * self.attn_scale
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, V)
        y = y.permute((0,4,1,2,3,5))
        y = y.reshape((B,Dq,H,W,C))
        y = self.proj(y)
        return y


class TemporalTransformer(nn.Module):
    def __init__(self, 
        channels,
        mlp_dim_mul=1,
        **kwargs
    ):
        super().__init__()
        self.attn1 = TemporalAttention(channels, **kwargs)
        self.attn2 = TemporalAttention(channels, **kwargs)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.mlp = MLP(channels, dim_mul=mlp_dim_mul)

    def forward(self, x, y):
        x = self.attn1(self.norm1(x)) + x # self attention
        x = self.attn2(self.norm2(x), y) + x # cross attention
        return self.mlp(self.norm3(x)) + x # feed-forward


class MLP(nn.Sequential):
    def __init__(self, dim, dim_mul=4):
        inner_dim = dim * dim_mul
        sequence = [
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, dim)
        ]
        super().__init__(*sequence)


def positional_encoding(position, dims, add_dims=()):
    div_term = torch.exp(
        torch.arange(0, dims, 2, device=position.device) * 
        (-math.log(10000.0) / dims)
    )
    if position.ndim == 1:    
        arg = position[:,None] * div_term[None,:]
    else:
        arg = position[:,:,None] * div_term[None,None,:]
    
    pos_enc = torch.concat(
        [torch.sin(arg), torch.cos(arg)],
        dim=-1
    )
    if add_dims:
        for dim in add_dims:
            pos_enc = pos_enc.unsqueeze(dim)
    return pos_enc
