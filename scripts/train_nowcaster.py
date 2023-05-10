import gc
import gzip
import os
import pickle

import numpy as np

from ldcast.features import batch, patches, split, transform

file_dir = os.path.dirname(os.path.abspath(__file__))


def setup_data(
    use_obs=True,
    use_nwp=True,
    obs_vars=("RZC",),
    nwp_vars=(
        "cape", "cin", "rate-cp", "rate-tp", "t2m",
        "tclw", "tcwv", "u", "v"
    ),
    nwp_lags=(0,12),
    target_var="RZC",
    batch_size=64,
    past_timesteps=4,
    future_timesteps=20,
    timestep_secs=300,
    nwp_timestep_secs=3600,
    sampler_file=None,
    chunks_file="../data/split_chunks.pkl.gz",
    sample_shape=(4,4)
):
    target = target_var + "-T"
    predictors_obs = [v + "-O" for v in obs_vars]
    predictors = []
    if use_obs:
        predictors += predictors_obs
    if use_nwp:
        predictors.append("nwp")
    
    variables = {
        target: {
            "sources": [target_var],
            "timesteps": np.arange(1,future_timesteps+1),
        }
    }
    for (var, raw_var) in zip(predictors_obs, obs_vars):
        variables[var] = {
            "sources": [raw_var],
            "timesteps": np.arange(-past_timesteps+1,1)
        }
    nwp_t1 = int(np.ceil(future_timesteps*timestep_secs/nwp_timestep_secs)) + 2
    nwp_range = np.arange(nwp_t1)
    variables["nwp"] = {
        "sources": nwp_vars,
        "timesteps": nwp_range,
        "timestep_secs": nwp_timestep_secs
    }

    # determine which raw variables are needed, then load them
    raw_vars = set.union(
        *(set(variables[v]["sources"]) for v in predictors_obs+[target])
    )
    for raw_var_base in variables["nwp"]["sources"]:
        raw_vars.update(f"{raw_var_base}-{lag}" for lag in nwp_lags)
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
    
    transform_rain = lambda: transform.default_rainrate_transform(
        raw["train"]["RZC"]["scale"]
    )
    transform_cape = lambda: transform.normalize_threshold(
        log=True,
        threshold=1.0, fill_value=1.0,
        mean=1.530, std=0.859
    )
    transform_rate_tp = lambda: transform.normalize_threshold(
        log=True,
        threshold=1e-5, fill_value=1e-5,
        mean=-3.831, std=0.650
    )
    transform_wind = lambda: transform.normalize(std=9.44)
    
    transforms = {
        "RZC-T": transform_rain(),
        "RZC-O": transform_rain(),
        "cape": transform_cape(),
        "cin": transform_cape(),
        "rate-tp": transform_rate_tp(),
        "rate-cp": transform_rate_tp(),
        "t2m": transform.normalize(mean=286.069, std=7.323),
        "tclw": transform.normalize_threshold(
            log=True,
            threshold=0.001, fill_value=0.001,
            mean=-1.486, std=0.638
        ),
        "tcwv": transform.normalize(std=17.307),        
        "u": transform_wind(),
        "v": transform_wind()
    }
    transforms["nwp"] = transform.combine([transforms[v] for v in nwp_vars])
    for (var_name, var_data) in variables.items():
        var_data["transform"] = transforms[var_name]
    
    if sampler_file is None:
        sampler_file = {
            "train": "../cache/sampler_nowcaster_train.pkl",
            "valid": "../cache/sampler_nowcaster_valid.pkl",
            "test": "../cache/sampler_nowcaster_test.pkl",
        }
    bins = np.exp(np.linspace(np.log(0.2), np.log(50), 10))

    datamodule = split.DataModule(
        variables, raw, predictors, target, target,
        forecast_raw_vars=nwp_vars,
        interval=timedelta(seconds=timestep_secs),
        batch_size=batch_size, sampling_bins=bins,
        time_range_sampling=(-past_timesteps+1,future_timesteps+1),
        sampler_file=sampler_file,
        sample_shape=sample_shape,
        valid_seed=1234, test_seed=2345,
    )
    
    gc.collect()
    return datamodule
