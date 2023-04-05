import concurrent
import multiprocessing
import os

import numpy as np

from ldcast.analysis import crps, histogram, rank, fss
from ldcast.features import io


all_models=(
    "mch-iters=50-res=256",
    "mch-dgmr",
    "mch-pysteps",
    "dwd-iters=50-res=256",
    "dwd-dgmr",
    "dwd-pysteps",
)


def histogram_all(models=all_models):
    os.makedirs("../results/histogram/", exist_ok=True)
    for model in models:
        if not model.startswith("mch"):
            continue
        ensembles_dir = f"../results/eval_ensembles/{model}-valid/"
        hist_fn = f"../results/histogram/{model}-valid.nc"
        histogram.save_histogram_for_dataset(ensembles_dir, hist_fn)


def prob_match_for_model(model):
    parts = model.split("-")
    parts = ["mch"] + parts[1:]
    model = "-".join(parts)

    hist_fn = f"../results/histogram/hist-{model}-valid.nc"
    (obs_hist, fc_hist, bins) = histogram.load_histogram(hist_fn)
    return histogram.probability_match_timesteps(obs_hist, fc_hist, bins)


def crps_all(models=all_models, prob_match=True, log=False):
    os.makedirs("../results/crps/", exist_ok=True)

    for model in models:
        pm = prob_match_for_model(model) if prob_match else None

        ensembles_dir = f"../results/eval_ensembles/{model}/"
        prefix = "logcrps" if log else "crps"
        if prob_match:
            prefix += "-pm"
        crps_fn = f"../results/crps/{prefix}-{model}.nc"

        crps.save_crps_for_dataset(
            ensembles_dir, crps_fn, log=log, preproc_fc=pm
        )


def ranks_all(models=all_models, prob_match=True):
    os.makedirs("../results/ranks/", exist_ok=True)

    for model in models:
        pm = prob_match_for_model(model) if prob_match else None

        ensembles_dir = f"../results/eval_ensembles/{model}/"
        prefix = "ranks"
        if prob_match:
            prefix += "-pm"
        ranks_fn = f"../results/ranks/{prefix}-{model}.nc"

        rank.save_ranks_for_dataset(
            ensembles_dir, ranks_fn, preproc_fc=pm
        )


def fractions_all(
    models=all_models,
    thresholds=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    prob_match=True
):
    os.makedirs("../results/fractions/", exist_ok=True)

    for model in models:
        pm = prob_match_for_model(model) if prob_match else None
        ensembles_dir = f"../results/eval_ensembles/{model}/"
        
        for T in thresholds:            
            prefix = "fractions"
            if prob_match:
                prefix = prefix + "-pm"
            prefix += f"-{T:.1f}"
            fss_fn = f"../results/fractions/{prefix}-{model}.nc"

            fss.save_fractions_for_dataset(
                ensembles_dir, fss_fn, T, preproc_fc=pm
            )


def rmse_ensemble_mean_batch(fn, log=False, pm=None):
    print(fn)
    (x, y, y_pred) = io.load_batch(fn, log=log, preproc_fc=pm)
    y_pred = y_pred.mean(axis=-1)
    return np.sqrt(((y-y_pred)**2).mean(axis=(1,3,4)))


def rmse_ensemble_mean(model="mch-iters=50-res=256", log=False, prob_match=True):
    ensembles_dir = f"../results/eval_ensembles/{model}/"
    files = os.listdir(ensembles_dir)
    files = (os.path.join(ensembles_dir,fn) for fn in sorted(files))
    
    pm = prob_match_for_model(model) if prob_match else None

    N_threads = multiprocessing.cpu_count()
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:
        tasks = [
            executor.submit(rmse_ensemble_mean_batch, fn, log, pm)
            for fn in files
        ]
        rmse = [task.result() for task in tasks]
    return np.concatenate(rmse, axis=0)
