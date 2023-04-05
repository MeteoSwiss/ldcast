import pytorch_lightning as pl
import torch

from ..diffusion import diffusion


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr=1e-4
):
    ldm = diffusion.LatentDiffusion(model, autoencoder, 
        context_encoder=context_encoder, lr=lr)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val_loss_ema", patience=6, verbose=True, check_finite=False
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping, checkpoint]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        strategy='dp' if (num_gpus > 1) else None,
        callbacks=callbacks,
        #precision=16
    )

    return (ldm, trainer)
