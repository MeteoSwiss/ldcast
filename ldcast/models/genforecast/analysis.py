import torch
from torch import nn
from torch.nn import functional as F

from ..nowcast.nowcast import AFNONowcastNetBase
from ..blocks.resnet import ResBlock3D


class AFNONowcastNetCascade(AFNONowcastNetBase):
    def __init__(self, *args, cascade_depth=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.cascade_depth = cascade_depth
        self.resnet = nn.ModuleList()        
        ch = self.embed_dim_out
        self.cascade_dims = [ch]
        for i in range(cascade_depth-1):
            ch_out = 2*ch
            self.cascade_dims.append(ch_out)
            self.resnet.append(
                ResBlock3D(ch, ch_out, kernel_size=(1,3,3), norm=None)
            )
            ch = ch_out

    def forward(self, x):
        x = super().forward(x)
        img_shape = tuple(x.shape[-2:])
        cascade = {img_shape: x}
        for i in range(self.cascade_depth-1):
            x = F.avg_pool3d(x, (1,2,2))
            x = self.resnet[i](x)
            img_shape = tuple(x.shape[-2:])
            cascade[img_shape] = x
        return cascade
