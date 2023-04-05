import numpy as np
import torch.nn as nn

from ..blocks.resnet import ResBlock3D
from ..utils import activation, normalization


class SimpleConvEncoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])
        
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i+1])
            res_kernel_size = (3,3,3) if i == 0 else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            downsample = nn.Conv3d(out_channels, out_channels,
                kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(downsample)
            in_channels = out_channels

        super().__init__(*sequence)


class SimpleConvDecoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i+1])
            out_channels = int(channels[i])
            upsample = nn.ConvTranspose3d(in_channels, in_channels, 
                    kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(upsample)
            res_kernel_size = (3,3,3) if (i == 0) else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            in_channels = out_channels

        super().__init__(*sequence)
