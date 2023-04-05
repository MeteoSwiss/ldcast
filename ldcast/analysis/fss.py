import concurrent.futures
import multiprocessing
import os

import netCDF4
import numpy as np

from ..features.io import load_batch, decode_saved_var_to_rainrate


def fractions_ensemble(observation, forecasts, threshold, max_scale=256):
    obs = (observation >= threshold).astype(np.float32)
    fc = (forecasts >= threshold).astype(np.float32).mean(axis=-1)
    obs_frac = {}
    fc_frac = {}
    
    scale = 1
    while True:
        obs_frac[scale] = obs.copy()
        fc_frac[scale] = fc.copy()
        scale *= 2
        if scale > max_scale:
            break
        obs = 0.25 * (
            obs[...,::2,::2] +
            obs[...,1::2,::2] +
            obs[...,::2,1::2] +
            obs[...,1::2,1::2]
        )
        fc = 0.25 * (
            fc[...,::2,::2] +
            fc[...,1::2,::2] +
            fc[...,::2,1::2] +
            fc[...,1::2,1::2]
        )
  
    return (obs_frac, fc_frac)


def frac_from_file(fn, threshold, preproc_fc):
    print(fn)
    (_, y, y_pred) = load_batch(fn, preproc_fc=preproc_fc)
    return fractions_ensemble(y, y_pred, threshold)


def save_fractions_for_dataset(data_dir, result_fn, threshold, preproc_fc=None):
    files = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir,fn) for fn in files]

    N_threads = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:
        futures = []
        for fn in files:
            args = (frac_from_file, fn, threshold, preproc_fc)
            futures.append(executor.submit(*args))

        (obs_frac, fc_frac) = zip(*(f.result() for f in futures))

    scales = list(obs_frac[0].keys())
    obs_frac_dict = {}
    fc_frac_dict = {}
    for s in scales:
        obs_frac_dict[s] = np.concatenate([f[s] for f in obs_frac], axis=0)
        fc_frac_dict[s] = np.concatenate([f[s] for f in fc_frac], axis=0)
    obs_frac = obs_frac_dict
    fc_frac = fc_frac_dict

    frac_vars = {}
    k = 0
    with netCDF4.Dataset(result_fn, 'w') as ds:
        ds.createDimension("dim_sample", obs_frac[1].shape[0])
        ds.createDimension("dim_channel", obs_frac[1].shape[1])
        ds.createDimension("dim_time_future", obs_frac[1].shape[2])
        var_params = {"zlib": True, "complevel": 1}
        for s in scales:
            ds.createDimension(f"dim_h_pool{s}x{s}", obs_frac[s].shape[3])
            ds.createDimension(f"dim_w_pool{s}x{s}", obs_frac[s].shape[4])
            obs_var = ds.createVariable(
                f"obs_frac_scale{s}x{s}", np.float32,
                (
                    "dim_sample", "dim_channel", "dim_time_future",
                    f"dim_h_pool{s}x{s}", f"dim_w_pool{s}x{s}",
                ),
                chunksizes=(1,)+obs_frac[s].shape[1:],
                **var_params
            )
            obs_var[:] = obs_frac[s]
            fc_var = ds.createVariable(
                f"fc_frac_scale{s}x{s}", np.float32,
                (
                    "dim_sample", "dim_channel", "dim_time_future",
                    f"dim_h_pool{s}x{s}", f"dim_w_pool{s}x{s}",
                ),
                chunksizes=(1,)+fc_frac[s].shape[1:],
                **var_params
            )
            fc_var[:] = fc_frac[s]


def load_fractions(fn):
    obs_frac = {}
    fc_frac = {}
    with netCDF4.Dataset(fn, 'r') as ds:
        var_list = ds.variables.keys()
        scales = {int(v.split("x")[-1]) for v in var_list}
        for s in scales:
            obs_frac[s] = np.array(ds[f"obs_frac_scale{s}x{s}"][:], copy=False)
            fc_frac[s] = np.array(ds[f"fc_frac_scale{s}x{s}"][:], copy=False)

    return (obs_frac, fc_frac)


def fractions_skill_score(
    obs_frac, fc_frac,
    frac_axes=None, fss_axes=None, use_timesteps=None
):
    if isinstance(obs_frac, dict):
        return {
            s: fractions_skill_score(
                obs_frac[s], fc_frac[s],
                frac_axes=frac_axes, fss_axes=fss_axes,
                use_timesteps=use_timesteps
            )
            for s in sorted(obs_frac)
        }

    if use_timesteps != None:
        obs_frac = obs_frac[:,:,:use_timesteps,...]
        fc_frac = fc_frac[:,:,:use_timesteps,...]
    fbs = ((obs_frac - fc_frac)**2).mean(axis=frac_axes)
    fbs_ref = (obs_frac**2).mean(axis=frac_axes) + \
        (fc_frac**2).mean(axis=frac_axes)
    fss = 1 - fbs/fbs_ref
    if isinstance(fss, np.ndarray):
        fss[~np.isfinite(fss)] = 1
        fss = fss.mean(axis=fss_axes)
    return fss
