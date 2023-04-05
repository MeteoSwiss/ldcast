import concurrent.futures
import multiprocessing
import os

import netCDF4
import numpy as np
from scipy.interpolate import interp1d

from ..features.io import load_batch, decode_saved_var_to_rainrate


def histogram(observation, forecasts, bins):
    N_bins = len(bins)-1
    N_timesteps = observation.shape[2]
    obs_hist = np.zeros((N_bins, N_timesteps), dtype=np.uint64)
    fc_hist = np.zeros((N_bins, N_timesteps), dtype=np.uint64)
    
    for t in range(observation.shape[2]):
        obs = observation[:,:,t,...].flatten()
        fc = forecasts[:,:,t,...].flatten()
        obs_hist[:,t] = np.histogram(obs, bins=bins)[0]
        fc_hist[:,t] = np.histogram(fc, bins=bins)[0]

    return (obs_hist, fc_hist)


def hist_from_file(fn, bins):
    print(fn)
    (_, y, y_pred) = load_batch(fn, threshold=bins[0])
    return histogram(y, y_pred, bins)


def save_histogram_for_dataset(data_dir, result_fn, bins=(0.05,120,100)):
    files = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir,fn) for fn in files]

    bins = np.exp(np.linspace(np.log(bins[0]), np.log(bins[1]), bins[2]))
    bins = np.hstack((0, bins))

    N_threads = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:
        futures = []
        for fn in files:
            args = (hist_from_file, fn, bins)
            futures.append(executor.submit(*args))

        (obs_hist, fc_hist) = zip(*(f.result() for f in futures))

    obs_hist = sum(obs_hist)
    fc_hist = sum(fc_hist)

    with netCDF4.Dataset(result_fn, 'w') as ds:
        ds.createDimension("dim_bin", obs_hist.shape[0])
        ds.createDimension("dim_time_future", obs_hist.shape[1])
        var_params = {"zlib": True, "complevel": 1}

        obs_var = ds.createVariable(
            f"obs_hist", np.uint64,
            ("dim_bin", "dim_time_future"),
            **var_params
        )
        obs_var[:] = obs_hist

        fc_var = ds.createVariable(
            f"fc_hist", np.uint64,
            ("dim_bin", "dim_time_future"),
            **var_params
        )
        fc_var[:] = fc_hist

        ds.createDimension("dim_bin_edge", len(bins))
        bin_var = ds.createVariable(
            f"bins", np.float64,
            ("dim_bin_edge",),
            **var_params
        )
        bin_var[:] = bins


def load_histogram(fn):
    with netCDF4.Dataset(fn, 'r') as ds:
        obs_hist = np.array(ds["obs_hist"][:], copy=False)
        fc_hist = np.array(ds["fc_hist"][:], copy=False)
        bins = np.array(ds["bins"][:], copy=False)

    return (obs_hist, fc_hist, bins)


class ProbabilityMatch:
    def __init__(self, obs_hist, fc_hist, bins):
        obs_c = obs_hist.cumsum()
        obs_c = obs_c / obs_c[-1]
        fc_c = fc_hist.cumsum()
        fc_c = fc_c / fc_c[-1]

        self.obs_cdf = interp1d(np.hstack((0,obs_c)), bins, fill_value='extrapolate')
        self.fc_cdf = interp1d(bins, np.hstack((0,fc_c)), fill_value='extrapolate')

    def __call__(self, x):
        return self.obs_cdf(self.fc_cdf(x))


def probability_match_timesteps(obs_hist, fc_hist, bins):
    num_timesteps = obs_hist.shape[1]
    return [
        ProbabilityMatch(obs_hist[:,t], fc_hist[:,t], bins)
        for t in range(num_timesteps)
    ]
