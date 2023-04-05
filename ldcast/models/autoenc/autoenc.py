import torch
from torch import nn
import pytorch_lightning as pl

from ..distributions import kl_from_standard_normal, ensemble_nll_normal
from ..distributions import sample_from_standard_normal


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self, 
        encoder, decoder, 
        kl_weight=0.01,     
        encoded_channels=64,
        hidden_width=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2*hidden_width,
            kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
            kernel_size=1)
        self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var)

    def _loss(self, batch):
        (x,y) = batch
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0][0]
        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (y-y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)

        total_loss = rec_loss + self.kl_weight * kl_loss

        return (total_loss, rec_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",
                "frequency": 1,
            },
        }
