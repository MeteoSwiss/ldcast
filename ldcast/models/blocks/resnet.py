from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

from ..utils import activation, normalization


class ResBlock3D(nn.Module):
    def __init__(
        self, in_channels, out_channels, resample=None,
        resample_factor=(1,1,1), kernel_size=(3,3,3), 
        act='swish', norm='group', norm_kwargs=None, 
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
        padding = tuple(k//2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)       
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode='trilinear')            
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2*p+s-k for (p,s,k) in zip(padding,resample_factor,kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor,
                padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)


    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in
