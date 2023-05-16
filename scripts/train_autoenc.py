import gc
import gzip
import os
import pickle

from fire import Fire
import numpy as np
from omegaconf import OmegaConf

from ldcast.features import batch, patches, split, transform
from ldcast.models.autoenc import encoder, training

file_dir = os.path.dirname(os.path.abspath(__file__))


def setup_data(
    var="RZC", 
    batch_size=64,
    sampler_file=None,
    num_timesteps=8,
    chunks_file="../data/split_chunks.pkl.gz"
):
    variables = {
        var: {
            "sources": [var],
            "timesteps": np.arange(num_timesteps),
        }
    }
    predictors = [var] # autoencoder: predictors == targets
    target = var
    raw_vars = [var]
    raw = {
        var: patches.load_all_patches(
            os.path.join(file_dir, f"../data/{var}/"), var
        )
        for var in raw_vars
    }

    # Load pregenerated train/valid/test split data.
    # These can be generated with features.split.get_chunks()
    with gzip.open(os.path.join(file_dir, chunks_file), 'rb') as f:
        chunks = pickle.load(f)
    (raw, _) = split.train_valid_test_split(raw, var, chunks=chunks)
    
    variables[var]["transform"] = transform.default_rainrate_transform(
        raw["train"][var]["scale"]
    )
    
    if sampler_file is None:
        sampler_file = {
            "train": "../cache/sampler_autoenc_train.pkl",
            "valid": "../cache/sampler_autoenc_valid.pkl",
            "test": "../cache/sampler_autoenc_test.pkl",
        }
    bins = np.exp(np.linspace(np.log(0.2), np.log(50), 10))
    datamodule = split.DataModule(
        variables, raw, predictors, target, var,
        sampling_bins=bins, batch_size=batch_size,
        sampler_file=sampler_file,
        valid_seed=1234, test_seed=2345
    )
    
    gc.collect()
    return datamodule


def setup_model(
    model_dir=None
):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    (autoencoder, trainer) = training.setup_autoenc_training(
        encoder=enc,
        decoder=dec,
        model_dir=model_dir
    )
    gc.collect()
    return (autoencoder, trainer)


def train(
    var="RZC", 
    batch_size=64,
    sampler_file=None,
    num_timesteps=8,
    chunks_file="../data/split_chunks.pkl.gz",
    model_dir=None,
    ckpt_path=None
):
    print("Loading data...")
    datamodule = setup_data(
        var=var, batch_size=batch_size, sampler_file=sampler_file,
        num_timesteps=num_timesteps, chunks_file=chunks_file
    )

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    train(**config)


if __name__ == "__main__":
    Fire(main)
