import gc

from fire import Fire
import torch
from omegaconf import OmegaConf

from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.genforecast import analysis, training, unet

from train_nowcaster import setup_data


def setup_model(
    num_timesteps=5,
    model_dir="../models/test/",
    autoenc_weights_fn="../models/autoenc/autoenc-32-0.01.pt",
    use_obs=True,
    use_nwp=False,
    nwp_input_patches=4,
    num_nwp_vars=9,
    lr=1e-4
):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()    
    autoencoder_obs = autoenc.AutoencoderKL(enc, dec)
    autoencoder_obs.load_state_dict(torch.load(autoenc_weights_fn))    

    autoencoders = []
    input_patches = []
    input_size_ratios = []
    embed_dim = []
    analysis_depth = []
    if use_obs:
        autoencoders.append(autoencoder_obs)
        input_patches.append(1)
        input_size_ratios.append(1)
        embed_dim.append(128)
        analysis_depth.append(4)
    if use_nwp:
        autoencoder_nwp = autoenc.DummyAutoencoder(width=num_nwp_vars)
        autoencoders.append(autoencoder_nwp)
        input_patches.append(nwp_input_patches)
        input_size_ratios.append(2)
        embed_dim.append(32)
        analysis_depth.append(2)

    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoders,
        input_patches=input_patches,
        input_size_ratios=input_size_ratios,
        train_autoenc=False,
        output_patches=num_timesteps,
        cascade_depth=3,
        embed_dim=embed_dim,
        analysis_depth=analysis_depth
    )

    model = unet.UNetModel(in_channels=autoencoder_obs.hidden_width,
        model_channels=256, out_channels=autoencoder_obs.hidden_width,
        num_res_blocks=2, attention_resolutions=(1,2), 
        dims=3, channel_mult=(1, 2, 4), num_heads=8,
        num_timesteps=num_timesteps, context_ch=analysis_net.cascade_dims
    )

    (ldm, trainer) = training.setup_genforecast_training(
        model, autoencoder_obs, context_encoder=analysis_net,
        model_dir=model_dir, lr=lr
    )
    gc.collect()
    return (ldm, trainer)


def train(
    future_timesteps=20,
    use_obs=True,
    use_nwp=False,
    sample_shape=(4,4),
    batch_size=64,
    sampler=None,
    ckpt_path=None,
    initial_weights=None,
    strict_weights=True,
    model_dir=None,
    lr=1e-4
):
    if sampler is None:
        sampler_file = None
    else:
        sampler_file = {
            s: f"{sampler}_{s}.pkl" for s in ["test", "train", "valid"]
        }

    print("Loading data...")
    datamodule = setup_data(
        future_timesteps=future_timesteps, use_obs=use_obs, use_nwp=use_nwp,
        sampler_file=sampler_file, batch_size=batch_size
    )

    print("Setting up model...")
    (model, trainer) = setup_model(
        num_timesteps=future_timesteps//4,
        use_obs=use_obs,
        use_nwp=use_nwp,
        model_dir=model_dir,
        lr=lr
    )
    if initial_weights is not None:
        print(f"Loading weights from {initial_weights}...")
        model.load_state_dict(
            torch.load(initial_weights, map_location=model.device),
            strict=strict_weights
        )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    train(**config)


if __name__ == "__main__":
    Fire(main)
