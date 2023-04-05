import os
import concurrent
import multiprocessing

import netCDF4
import numpy as np

from ..features.io import load_batch


def ranks_ensemble(
    observation, forecasts,
    noise_scale=1e-6, rng=None
):
    shape = observation.shape
    N = np.prod(shape)
    shape_flat = (np.prod(shape),)    
    observation = observation.reshape((N,))
    forecasts = forecasts.reshape((N, forecasts.shape[-1]))    
    N_threads = multiprocessing.cpu_count()

    max_rank = forecasts.shape[-1]
    bins = np.arange(-0.5, max_rank+0.6)
    ranks_all = np.zeros_like(observation, dtype=np.uint32)

    if rng is None:
        rng = np.random

    def rank_dist_chunk(k):
        i0 = int(round((k/N_threads) * N))
        i1 = int(round(((k+1) / N_threads) * N))
        obs = observation[i0:i1].astype(np.float64, copy=True)
        fc = forecasts[i0:i1,:].astype(np.float64, copy=True)

        # add a tiny amount of noise to forecast to randomize ties
        # (important to add to both obs and fc!)
        obs += (rng.rand(*obs.shape) - 0.5) * noise_scale
        fc += (rng.rand(*fc.shape) - 0.5) * noise_scale

        ranks = np.count_nonzero(obs[...,None] >= fc, axis=-1)        
        ranks_all[i0:i1] = ranks

    with concurrent.futures.ThreadPoolExecutor(N_threads) as executor:
        futures = {}
        for k in range(N_threads):
            args = (rank_dist_chunk, k)
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)

    return ranks_all.reshape(shape)


def ranks_multiscale(observation, forecasts):
    obs = observation
    fc = forecasts

    rank_scales = {}
    scale = 1
    while True:
        r = ranks_ensemble(obs, fc)
        rank_scales[scale] = r
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
    
    return rank_scales


def rank_distribution(ranks, num_forecasts=32):
    N = np.prod(ranks.shape)
    bins = np.arange(-0.5, num_forecasts+0.6)
    N_threads = multiprocessing.cpu_count()
    ranks = ranks.ravel()
    
    hist = [None] * N_threads
    def hist_chunk(k):
        i0 = int(round((k/N_threads) * N))
        i1 = int(round(((k+1) / N_threads) * N))
        (h, _) = np.histogram(ranks[i0:i1], bins=bins)
        hist[k] = h

    with concurrent.futures.ThreadPoolExecutor(N_threads) as executor:
        futures = {}
        for k in range(N_threads):
            args = (hist_chunk, k)
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)    

    hist = sum(hist)
    return hist / hist.sum()


def rank_KS(rank_dist, num_forecasts=32):
    h = rank_dist
    h = h / h.sum()
    ch = np.cumsum(h)
    cb = np.linspace(0, 1, len(ch))
    return abs(ch-cb).max()


def rank_DKL(rank_dist, num_forecasts=32):
    h = rank_dist
    q = h / h.sum()
    p = 1/len(h)
    return p*np.log(p/q).sum()


def rank_metric_by_leadtime(ranks, metric=None, num_forecasts=32):
    if metric is None:
        metric = rank_DKL

    metric_by_leadtime = []
    for t in range(ranks.shape[2]):
        ranks_time = ranks[:,:,t,...]
        h = rank_distribution(ranks_time)
        m = metric(h, num_forecasts=num_forecasts)
        metric_by_leadtime.append(m)
    return np.array(metric_by_leadtime)


def rank_metric_by_bin(ranks, values, bins, metric=None, num_forecasts=32):
    if metric is None:
        metric = rank_DKL

    metric_by_bin = []
    for (b0,b1) in zip(bins[:-1],bins[1:]):
        ranks_bin = ranks[(b0 <= values) & (values < b1)]
        h = rank_distribution(ranks_bin)
        m = metric(h, num_forecasts=num_forecasts)
        metric_by_bin.append(m)
    return np.array(metric_by_bin)


def process_batch(fn, preproc_fc=None):
    print(fn)
    (_, y, y_pred) = load_batch(fn, preproc_fc=preproc_fc)
    return ranks_multiscale(y, y_pred)


def save_ranks_for_dataset(data_dir, result_fn, preproc_fc=None):
    files = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir,fn) for fn in files]

    N_threads = multiprocessing.cpu_count()
    futures = []
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:        
        for fn in files:
            args = (process_batch, fn)
            kwargs = {"preproc_fc": preproc_fc}
            futures.append(executor.submit(*args, **kwargs))
        
    ranks = [f.result() for f in futures]
    scales = sorted(ranks[0].keys())
    ranks = {
        s: np.concatenate([r[s] for r in ranks], axis=0)
        for s in scales
    }

    with netCDF4.Dataset(result_fn, 'w') as ds:
        ds.createDimension("dim_sample", ranks[1].shape[0])
        ds.createDimension("dim_channel", ranks[1].shape[1])
        ds.createDimension("dim_time_future", ranks[1].shape[2])
        var_params = {"zlib": True, "complevel": 1}

        for s in scales:
            ds.createDimension(f"dim_h_pool{s}x{s}", ranks[s].shape[3])
            ds.createDimension(f"dim_w_pool{s}x{s}", ranks[s].shape[4])
            var = ds.createVariable(
                f"ranks_pool{s}x{s}", np.float32,
                (
                    "dim_sample", "dim_channel", "dim_time_future",
                    f"dim_h_pool{s}x{s}", f"dim_w_pool{s}x{s}",
                ),
                chunksizes=(1,)+ranks[s].shape[1:],
                **var_params
            )
            var[:] = ranks[s]
