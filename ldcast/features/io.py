import os

import netCDF4
import numpy as np


def convert_var_for_saving(
    x, fill_value=0.02, min_value=0.05, max_value=118.428,
    mean=-0.051, std=0.528
):
    y = x*std + mean    
    log_min = np.log10(min_value)
    log_max = np.log10(max_value)
    mask = (y >= log_min)
    y = y[mask].clip(max=log_max)
    y = (y-log_min) / (log_max-log_min)
    yc = np.zeros_like(x, dtype=np.uint16)
    yc[mask] = (y*65533).round().astype(np.uint16) + 1
    return yc


def decode_saved_var_to_rainrate(
    x, fill_value=0.02, min_value=0.05, threshold=0.1, max_value=118.428,
    mean=-0.051, std=0.528, log=False, preproc=None
):
    mask = (x >= 1)
    log_min = np.log10(min_value)
    log_max = np.log10(max_value)
    yc = log_min + (x[mask].astype(np.float32)-1) * \
        ((log_max-log_min) / 65533)
    y = np.zeros_like(x, dtype=np.float32)

    yc = 10**yc        
    y[mask] = yc
    if preproc is not None:
        y = [preproc[t](y[:,:,t,...]) for t in range(y.shape[2])]
        y = np.stack(y, axis=2)
    
    if log:
        y[y < threshold] = fill_value
        y = np.log10(y)
    else:
        y[y < threshold] = 0.0
    
    return y


def save_batch(x, y, y_pred, batch_index, fn_template, out_dir, out_fn=None):
    while isinstance(x, list) or isinstance(x, tuple):
        x = x[0]

    x = convert_var_for_saving(np.array(x, copy=False))
    y = convert_var_for_saving(np.array(y, copy=False))
    y_pred = convert_var_for_saving(np.array(y_pred, copy=False))

    if out_fn is None:
        out_fn = fn_template.format(batch_index=batch_index)
    out_fn = os.path.join(out_dir, out_fn)

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_sample = ds.createDimension("dim_sample", y.shape[0])
        dim_channel = ds.createDimension("dim_channel", y.shape[1])
        dim_time_past = ds.createDimension("dim_time_past", x.shape[2])
        dim_time_future = ds.createDimension("dim_time_future", y.shape[2])
        dim_h = ds.createDimension("dim_h", y.shape[3])
        dim_w = ds.createDimension("dim_w", y.shape[4])
        dim_member = ds.createDimension("dim_member", y_pred.shape[5])
        var_params = {"zlib": True, "complevel": 1}

        var_fc = ds.createVariable(
            "forecasts", y_pred.dtype,
            (
                "dim_sample", "dim_channel",
                "dim_time_future", "dim_h", "dim_w", "dim_member"
            ),
            **var_params
        )
        var_fc[:] = y_pred

        var_obs_past = ds.createVariable(
            "past_observations", x.dtype,
            ("dim_sample", "dim_channel", "dim_time_past", "dim_h", "dim_w"),
            **var_params
        )
        var_obs_past[:] = x

        var_obs_future = ds.createVariable(
            "future_observations", y.dtype,
            ("dim_sample", "dim_channel", "dim_time_future", "dim_h", "dim_w"),
            **var_params
        )
        var_obs_future[:] = y


def load_batch(fn, decode=True, preproc_fc=None, **kwargs):
    with netCDF4.Dataset(fn, 'r') as ds:
        y_pred = np.array(ds["forecasts"][:], copy=False)
        x = np.array(ds["past_observations"][:], copy=False)
        y = np.array(ds["future_observations"][:], copy=False)

    if decode:
        x = decode_saved_var_to_rainrate(x, **kwargs)
        y = decode_saved_var_to_rainrate(y, **kwargs)
        y_pred = decode_saved_var_to_rainrate(
            y_pred, preproc=preproc_fc, **kwargs
        )

    return (x, y, y_pred)


def load_all_observations(
    ensemble_dir, decode=True, preproc_fc=None,
    timeframe='future', **kwargs
):
    files = os.listdir(ensemble_dir)
    obs = []
    for fn in sorted(files):
        with netCDF4.Dataset(os.path.join(ensemble_dir, fn), 'r') as ds:
            var = f"{timeframe}_observations"
            x = np.array(ds[var][:], copy=False)
            x = decode_saved_var_to_rainrate(x, **kwargs)
            obs.append(x)

    obs = np.concatenate(obs, axis=0)
    return obs
