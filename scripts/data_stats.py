from datetime import datetime, timedelta
import gzip
import os
import pickle

import dask
import netCDF4
import numpy as np


def rainrate_hist(
    data_dir="../data/RZC",
    bins=(0, 0.1, 1.0, 4.0, 5.0, 8.0, 10.0, np.inf),
):
    files = os.listdir(data_dir)
    files = (os.path.join(data_dir, fn) for fn in files)
    @dask.delayed
    def hist_file(fn):
        print(fn)
        with netCDF4.Dataset(fn, 'r') as ds:
            patches = np.array(ds["patches"][:], copy=False)
            if "scale" in ds.variables:
                scale = np.array(ds["scale"][:], copy=False)
                patches = scale[patches]
            zero_patch_vol = ds["zero_patch_times"].shape[0] * (
                patches.shape[1] * patches.shape[2] )

        patches[~np.isfinite(patches)] = 0.0        
        (hist_patch, _) = np.histogram(patches.ravel(), bins=bins)
        hist_patch[0] += zero_patch_vol
        return hist_patch

    return sum(dask.compute(
        [hist_file(fn) for fn in files], 
        scheduler='processes'
    )[0])


def split_dates(out_fn, split="train", months=(4,5,6,7,8,9)):
    def get_date(t):
        return datetime(1970,1,1) + timedelta(seconds=int(t))

    with gzip.open("../data/split_chunks.pkl.gz", 'rb') as f:
        chunks = pickle.load(f)

    dates = chunks[split]

    with open(out_fn, 'w') as f:
        for (t0,t1) in dates:
            dt = get_date(t0)
            dt1 = get_date(t1)
            print(dt, dt1)
            while dt < dt1:
                if dt.month in months:
                    print(dt.strftime("%Y-%m-%d"), file=f)
                dt += timedelta(days=1)
