import collections

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..blocks.afno import AFNOBlock3d
from ..blocks.attention import positional_encoding, TemporalTransformer


class Nowcaster(pl.LightningModule):
    def __init__(self, nowcast_net):
        super().__init__()
        self.nowcast_net = nowcast_net

    def forward(self, x):
        return self.nowcast_net(x)

    def _loss(self, batch):
        (x,y) = batch
        y_pred = self.forward(x)
        return (y-y_pred).square().mean()

    def training_step(self, batch, batch_idx):        
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )

        optimizer_spec = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
        return optimizer_spec


class AFNONowcastNetBasic(nn.Sequential):
    def __init__(
        self,
        embed_dim=256,
        depth=12,
        patch_size=(4,4,4)
    ):
        patch_embed = PatchEmbed3d(
            embed_dim=embed_dim, patch_size=patch_size
        )
        blocks = nn.Sequential(
            *(AFNOBlock(embed_dim) for _ in range(depth))
        )
        patch_expand = PatchExpand3d(
            embed_dim=embed_dim, patch_size=patch_size
        )
        super().__init__(*[patch_embed, blocks, patch_expand])


class FusionBlock3d(nn.Module):
    def __init__(self, dim, size_ratios, dim_out=None, afno_fusion=False):
        super().__init__()

        N_sources = len(size_ratios)
        if not isinstance(dim, collections.abc.Sequence):
            dim = (dim,) * N_sources
        if dim_out is None:
            dim_out = dim[0]
        
        self.scale = nn.ModuleList()
        for (i,size_ratio) in enumerate(size_ratios):
            if size_ratio == 1:
                scale = nn.Identity()
            else:
                scale = []
                while size_ratio > 1:
                    scale.append(nn.ConvTranspose3d(
                        dim[i], dim_out if size_ratio==2 else dim[i],
                        kernel_size=(1,3,3), stride=(1,2,2),
                        padding=(0,1,1), output_padding=(0,1,1)
                    ))
                    size_ratio //= 2
                scale = nn.Sequential(*scale)
            self.scale.append(scale)

        self.afno_fusion = afno_fusion
        
        if self.afno_fusion:
            if N_sources > 1:
                self.fusion = nn.Sequential(
                    nn.Linear(sum(dim), sum(dim)),
                    AFNOBlock3d(dim*N_sources, mlp_ratio=2),
                    nn.Linear(sum(dim), dim_out)
                )
            else:
                self.fusion = nn.Identity()
        
    def resize_proj(self, x, i):
        x = x.permute(0,4,1,2,3)
        x = self.scale[i](x)
        x = x.permute(0,2,3,4,1)
        return x

    def forward(self, x):
        x = [self.resize_proj(xx, i) for (i, xx) in enumerate(x)]
        if self.afno_fusion:        
            x = torch.concat(x, axis=-1)
            x = self.fusion(x)
        else:
            x = sum(x)
        return x


class AFNONowcastNetBase(nn.Module):
    def __init__(
        self,
        autoencoder,
        embed_dim=128,
        embed_dim_out=None,
        analysis_depth=4,
        forecast_depth=4,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=2,
        train_autoenc=False,
        afno_fusion=False
    ):
        super().__init__()
        
        self.train_autoenc = train_autoenc
        if not isinstance(autoencoder, collections.abc.Sequence):
            autoencoder = [autoencoder]
        if not isinstance(input_patches, collections.abc.Sequence):
            input_patches = [input_patches]        
        num_inputs = len(autoencoder)
        if not isinstance(embed_dim, collections.abc.Sequence):
            embed_dim = [embed_dim] * num_inputs
        if embed_dim_out is None:
            embed_dim_out = embed_dim[0]
        if not isinstance(analysis_depth, collections.abc.Sequence):
            analysis_depth = [analysis_depth] * num_inputs
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_patches = output_patches

        # encoding + analysis for each input
        self.autoencoder = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.analysis = nn.ModuleList()
        for i in range(num_inputs):
            ae = autoencoder[i].requires_grad_(train_autoenc)
            self.autoencoder.append(ae)

            proj = nn.Conv3d(ae.hidden_width, embed_dim[i], kernel_size=1)
            self.proj.append(proj)

            analysis = nn.Sequential(
                *(AFNOBlock3d(embed_dim[i]) for _ in range(analysis_depth[i]))
            )
            self.analysis.append(analysis)

        # temporal transformer
        self.use_temporal_transformer = \
            any((ipp != output_patches) for ipp in input_patches)
        if self.use_temporal_transformer:
            self.temporal_transformer = nn.ModuleList(
                TemporalTransformer(embed_dim[i]) for i in range(num_inputs)
            )

        # data fusion
        self.fusion = FusionBlock3d(embed_dim, input_size_ratios,
            afno_fusion=afno_fusion, dim_out=embed_dim_out)

        # forecast
        self.forecast = nn.Sequential(
            *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
        )

    def add_pos_enc(self, x, t):
        if t.shape[1] != x.shape[1]:
            # this can happen if x has been compressed 
            # by the autoencoder in the time dimension
            ds_factor = t.shape[1] // x.shape[1]
            t = F.avg_pool1d(t.unsqueeze(1), ds_factor)[:,0,:]

        pos_enc = positional_encoding(t, x.shape[-1], add_dims=(2,3))
        return x + pos_enc

    def forward(self, x):
        (x, t_relative) = list(zip(*x))

        # encoding + analysis for each input
        def process_input(i):
            z = self.autoencoder[i].encode(x[i])[0]
            z = self.proj[i](z)
            z = z.permute(0,2,3,4,1)
            z = self.analysis[i](z)
            if self.use_temporal_transformer:
                # add positional encoding
                z = self.add_pos_enc(z, t_relative[i])
                
                # transform to output shape and coordinates
                expand_shape = z.shape[:1] + (-1,) + z.shape[2:]
                pos_enc_output = positional_encoding(
                    torch.arange(1,self.output_patches+1, device=z.device), 
                    self.embed_dim[i], add_dims=(0,2,3)
                )
                pe_out = pos_enc_output.expand(*expand_shape)
                z = self.temporal_transformer[i](pe_out, z)
            return z

        x = [process_input(i) for i in range(len(x))]
        
        # merge inputs
        x = self.fusion(x)
        # produce prediction
        x = self.forecast(x)
        return x.permute(0,4,1,2,3) # to channels-first order


class AFNONowcastNet(AFNONowcastNetBase):
    def __init__(self, autoencoder, output_autoencoder=None, **kwargs):
        super().__init__(autoencoder, **kwargs)
        if output_autoencoder is None:
            output_autoencoder = autoencoder[0]
        self.output_autoencoder = output_autoencoder.requires_grad_(
            self.train_autoenc)
        self.out_proj = nn.Conv3d(
            self.embed_dim_out, output_autoencoder.hidden_width, kernel_size=1
        )

    def forward(self, x):
        x = super().forward(x)
        x = self.out_proj(x)
        return self.output_autoencoder.decode(x)
