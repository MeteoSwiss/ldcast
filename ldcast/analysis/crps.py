import concurrent.futures
import multiprocessing
import os

import netCDF4
import numpy as np

from ..features.io import load_batch, decode_saved_var_to_rainrate


def crps_ensemble(observation, forecasts):
    shape = observation.shape
    N = np.prod(shape)
    shape_flat = (np.prod(shape),)
    observation = observation.reshape((N,))
    forecasts = forecasts.reshape((N, forecasts.shape[-1]))
    crps_all = np.zeros_like(observation)
    N_threads = multiprocessing.cpu_count()

    def crps_chunk(k):
        i0 = int(round((k/N_threads) * N))
        i1 = int(round(((k+1) / N_threads) * N))
        obs = observation[i0:i1].copy()
        fc = forecasts[i0:i1,:].copy()
        fc.sort(axis=-1)
        fc_below = fc < obs[...,None]
        crps = np.zeros_like(obs)
            
        for i in range(fc.shape[-1]):
            below = fc_below[...,i]
            weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
            crps[below] += weight * (obs[below]-fc[...,i][below])

        for i in range(fc.shape[-1]-1,-1,-1):
            above = ~fc_below[...,i]
            k = fc.shape[-1]-1-i
            weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
            crps[above] += weight * (fc[...,i][above]-obs[above])

        crps_all[i0:i1] = crps

    with concurrent.futures.ThreadPoolExecutor(N_threads) as executor:
        futures = {}
        for k in range(N_threads):
            args = (crps_chunk, k)
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)

    return crps_all.reshape(shape)


def crps_ensemble_multiscale(observation, forecasts):
    obs = observation
    fc = forecasts

    crps_scales = {}
    scale = 1
    while True:
        c = crps_ensemble(obs, fc)
        crps_scales[scale] = c
        scale *= 2
        if obs.shape[-1] == 1:
            break
        # avg pooling
        obs = 0.25 * (
            obs[...,::2,::2] +
            obs[...,1::2,::2] +
            obs[...,::2,1::2] +
            obs[...,1::2,1::2]
        )
        fc = 0.25 * (
            fc[...,::2,::2,:] +
            fc[...,1::2,::2,:] +
            fc[...,::2,1::2,:] +
            fc[...,1::2,1::2,:]
        )
    
    return crps_scales


def gather_observation(data_dir):
    files = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir,fn) for fn in files]

    def obs_from_file(fn):
        with netCDF4.Dataset(fn, 'r') as ds:
            obs = np.array(ds["future_observations"][:], copy=False)
        obs = decode_saved_var_to_rainrate(obs)
        p = 1
        obs_pooled = {}
        while True:
            obs_pooled[p] = obs
            if obs.shape[-1] == 1:
                break
            obs = 0.25 * (
                obs[...,::2,::2] +
                obs[...,1::2,::2] +
                obs[...,::2,1::2] +
                obs[...,1::2,1::2]
            )
            p *= 2
        return obs_pooled

    obs_pooled = {}
    for fn in files:
        print(fn)
        obs_file = obs_from_file(fn)
        for k in obs_file:
            if k not in obs_pooled:
                obs_pooled[k] = []
            obs_pooled[k].append(obs_file[k])
    
    for k in obs_pooled:
        obs_pooled[k] = np.concatenate(obs_pooled[k], axis=0)

    return obs_pooled


def process_batch(fn, log=False, preproc_fc=None):
    print(fn)
    (_, y, y_pred) = load_batch(fn, log=log, preproc_fc=preproc_fc)
    return crps_ensemble_multiscale(y, y_pred)


def save_crps_for_dataset(data_dir, result_fn, log=False, preproc_fc=None):
    files = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir,fn) for fn in files]

    N_threads = multiprocessing.cpu_count()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:        
        for fn in files:
            args = (process_batch, fn)
            kwargs = {"log": log, "preproc_fc": preproc_fc}
            futures.append(executor.submit(*args, **kwargs))
        
    crps = [f.result() for f in futures]
    scales = sorted(crps[0].keys())
    crps = {
        s: np.concatenate([c[s] for c in crps], axis=0)
        for s in scales
    }

    with netCDF4.Dataset(result_fn, 'w') as ds:
        ds.createDimension("dim_sample", crps[1].shape[0])
        ds.createDimension("dim_channel", crps[1].shape[1])
        ds.createDimension("dim_time_future", crps[1].shape[2])
        var_params = {"zlib": True, "complevel": 1}

        for s in scales:
            ds.createDimension(f"dim_h_pool{s}x{s}", crps[s].shape[3])
            ds.createDimension(f"dim_w_pool{s}x{s}", crps[s].shape[4])
            var = ds.createVariable(
                f"crps_pool{s}x{s}", np.float32,
                (
                    "dim_sample", "dim_channel", "dim_time_future",
                    f"dim_h_pool{s}x{s}", f"dim_w_pool{s}x{s}",
                ),
                chunksizes=(1,)+crps[s].shape[1:],
                **var_params
            )
            var[:] = crps[s]
