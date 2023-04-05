from datetime import datetime, timedelta
import os

import dask
import netCDF4
import numpy as np

from .utils import average_pool


def patch_locations(
    time_range,
    patch_box,
    patch_shape=(32,32),
    interval=timedelta(minutes=5),
    epoch=(1970,1,1)
):
    patches = {}
    t = time_range[0]
    while t < time_range[1]:
        patches[t] = []
        for pi in range(patch_box[0][0], patch_box[0][1]):
            for pj in range(patch_box[1][0], patch_box[1][1]):
                patches[t].append((pi,pj))
        patches[t] = np.array(patches[t])
        t += interval

    return patches


def save_patches_radar(
    patches, archive_path, out_dir,
    variables=("RZC", "CPCH"),
    suffix="2020",
    **kwargs
):
    from ..datasets import mchradar

    source_vars = {}
    mchradar_reader = mchradar.MCHRadarReader(
        archive_path=archive_path,
        variables=variables,
        phys_values=False
    )

    ezc_nonzero_count_func = lambda x: np.count_nonzero((x >= 1) & (x<251))
    nonzero_count_func = {
        "RZC": lambda x: np.count_nonzero(x > 1),
        "CPCH": lambda x: np.count_nonzero(x > 1)
    }
    zero_value = {v: 0 for v in variables}    
    zero_value["RZC"] = 1
    zero_value["CPCH"] = 1

    save_patches_all(
        mchradar_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        source_vars=source_vars, min_nonzeros_to_include=5,
        **kwargs
    )


def save_patches_dwdradar(
    patches, archive_path, out_dir,
    variables=("RV",),
    suffix="2022",
    patch_shape=(32,32),
    **kwargs
):
    from ..datasets import dwdradar

    source_vars = {}
    dwdradar_reader = dwdradar.DWDRadarReader(
        archive_path=archive_path,
        variables=variables
    )

    patches_flt = {}
    for t in sorted(patches):
        if (t.hour==0) and (t.minute==0):
                print(t)

        try:
            data = dwdradar_reader.variable_for_time(t, "RV")
        except FileNotFoundError:
            continue

        patch_locs_time = []
        for (pi,pj) in patches[t]:
            i0 = pi * patch_shape[0]
            i1 = i0 + patch_shape[0]
            j0 = pj * patch_shape[1]
            j1 = j0 + patch_shape[1]
            patch = data[i0:i1,j0:j1]
            if np.isfinite(patch).all():
                patch_locs_time.append((pi,pj))
        
        if patch_locs_time:
            patches_flt[t] = np.array(patch_locs_time)
    
    print(len(patches),len(patches_flt))
    patches = patches_flt

    nonzero_count_func = {"RV": np.count_nonzero}
    zero_value = {v: 0 for v in variables}

    save_patches_all(
        dwdradar_reader, patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix,
        source_vars=source_vars, min_nonzeros_to_include=5,
        **kwargs
    )


def save_patches_ifs(
    patches, archive_path, out_dir,
    variables=(
        #"rate-tp", "rate-cp", "t2m", "cape", "cin", 
        #"tclw", "tcwv", #, "rate-tpe"
        #"u", "v"
        "cin",
    ),
    suffix="2020",
    lags=(0,12)
):
    from ..datasets import ifsnwp
    from .. import projection

    proj = projection.GridProjection(projection.ccs4_swiss_grid_area)
    ifsnwp_reader = ifsnwp.IFSNWPReader(
        proj,
        archive_path=archive_path,
        variables=variables,        
        lags=lags,
    )

    # we only get data for every hour, so modify patches
    ifs_patches = {
        dt: pset for (dt, pset) in patches.items()
        if (dt.minute == dt.second == dt.microsecond == 0)
    }

    variables_with_lag = []
    for lag in ifsnwp_reader.lags:
        variables_with_lag.extend(f"{v}-{lag}" for v in variables)

    count_positive = lambda x: np.count_nonzero(x > 0)
    all_nonzero = lambda x: np.prod(x.shape)
    nonzero_count_func = {
        "rate-tp": count_positive,
        "rate-cp": count_positive,
        "t2m": all_nonzero,        
        "cape": count_positive,
        "cin": count_positive,
        "tclw": count_positive,
        "tcwv": count_positive,
        "u": all_nonzero,
        "v": all_nonzero,
        "rate-tpe": count_positive,
    }    
    nonzero_count_func = {
        v: nonzero_count_func[v.rsplit("-", 1)[0]]
        for v in variables_with_lag    
    }
    postproc = {
        f"cin-{lag}": lambda x: np.nan_to_num(x, nan=0.0, copy=False)
        for lag in lags
    }
    zero_value = {v: 0 for v in variables_with_lag}
    avg_pool = lambda x: average_pool(x, factor=8, missing=np.nan)
    pool = {v: avg_pool for v in variables_with_lag}

    save_patches_all(ifsnwp_reader, ifs_patches, variables_with_lag,
        nonzero_count_func, zero_value, out_dir, suffix, pool=pool,
        postproc=postproc)


def save_patches_cosmo(patches, archive_path, out_dir, suffix="2020"):
    from ..datasets import cosmonwp

    cosmonwp_reader = cosmonwp.COSMOCCS4Reader(
        archive_path=archive_path, cache_size=6000)

    # we only get data for every hour, so modify patches
    cosmo_patches = {}
    for (dt,pset) in patches.items():
        dt0 = datetime(dt.year, dt.month, dt.day, dt.hour)
        dt1 = dt0 + timedelta(hours=1)
        if dt0 not in cosmo_patches:
            cosmo_patches[dt0] = set()
        if dt1 not in cosmo_patches:
            cosmo_patches[dt1] = set()
        cosmo_patches[dt0].update(pset)
        cosmo_patches[dt1].update(pset)

    variables = [
        "CAPE_MU", "CIN_MU", "SLI", 
        "HZEROCL", "LCL_ML", "MCONV", "OMEGA",
        "T_2M", "T_SO", "SOILTYP"
    ]
    count_positive = lambda x: np.count_nonzero(x>0)
    all_nonzero = lambda x: np.prod(x.shape)
    nonzero_count_func = {
        "CAPE_MU": count_positive,
        "CIN_MU": count_positive,
        "SLI": all_nonzero,        
        "HZEROCL": count_positive,
        "LCL_ML": count_positive,
        "MCONV": all_nonzero,
        "OMEGA": all_nonzero,
        "T_2M": all_nonzero,
        "T_SO": all_nonzero,
        "SOILTYP": lambda x: np.count_nonzero(x!=5)
    }
    zero_value = {v: 0 for v in variables}
    zero_value["SOILTYP"] = 5

    save_patches_all(cosmonwp_reader, cosmo_patches, variables,
        nonzero_count_func, zero_value, out_dir, suffix, pool=pool)


def save_patches_all(
    reader, patches, variables, nonzero_count_func, zero_value,
    out_dir, suffix, epoch=datetime(1970,1,1), postproc={}, scale=None,
    pool={}, source_vars={}, parallel=False, min_nonzeros_to_include=1
):

    def save_var(var_name):
        src_name = source_vars.get(var_name, var_name)

        (patch_data, patch_coords, patch_times, 
            zero_patch_coords, zero_patch_times) = get_patches(
                reader, src_name, patches, 
                nonzero_count_func=nonzero_count_func[var_name],
                postproc=postproc.get(var_name),
                pool=pool.get(var_name)
            )
        try:
            time = epoch + timedelta(seconds=int(patch_times[0]))
            var_scale = reader.get_scale(time, var_name)
        except (AttributeError, KeyError):
            var_scale = None if (scale is None) else scale[var_name]
            pass
        
        var_name = var_name.replace("_", "-")
        out_fn = f"patches_{var_name}_{suffix}.nc"
        out_path = os.path.join(out_dir, var_name)
        os.makedirs(out_path, exist_ok=True)
        out_fn = os.path.join(out_path, out_fn)
        
        save_patches(
            patch_data, patch_coords, patch_times,
            zero_patch_coords, zero_patch_times, out_fn,
            zero_value=zero_value[var_name], scale=var_scale
        )

    if parallel:
        save_var = dask.delayed(save_var)

    jobs = [save_var(v) for v in variables]
    if parallel:
        dask.compute(jobs, scheduler='threads')


def get_patches(
    reader, variable, patches,
    patch_shape=(32,32), nonzero_count_func=None,
    epoch=datetime(1970,1,1), postproc=None,
    pool=None, min_nonzeros_to_include=1
):
    num_patches = sum(len(patches[t]) for t in patches)
    patch_data = []
    patch_coords = []
    patch_times = []
    zero_patch_coords = []
    zero_patch_times = []
    
    if hasattr(reader, "phys_values"):
        phys_values = reader.phys_values
    
    k = 0
    try:
        if hasattr(reader, "phys_values"):
            reader.phys_values = False
        for (t, p_coord) in patches.items():
            try:
                data = reader.variable_for_time(t, variable)
            except (ValueError, FileNotFoundError, KeyError, OSError):
                continue

            if postproc is not None:
                data = postproc(data)

            time_sec = np.int64((t-epoch).total_seconds())
            for (pi, pj) in p_coord:
                if k % 100000 == 0:
                    print("{}: {}/{}".format(t, k, num_patches))
                patch_box = data[
                    pi*patch_shape[0]:(pi+1)*patch_shape[0],
                    pj*patch_shape[1]:(pj+1)*patch_shape[1],
                ].copy()
                is_nonzero = (nonzero_count_func is not None) and \
                    (nonzero_count_func(patch_box) < min_nonzeros_to_include)
                if is_nonzero:
                    zero_patch_coords.append((pi,pj))
                    zero_patch_times.append(time_sec)
                else:
                    if pool is not None:
                        patch_box = pool(patch_box)
                    patch_data.append(patch_box)
                    patch_coords.append((pi,pj))
                    patch_times.append(time_sec)
                k += 1
                
    finally:
        if hasattr(reader, "phys_values"):
            reader.phys_values = phys_values

    if zero_patch_coords:
        zero_patch_coords = np.stack(zero_patch_coords, axis=0).astype(np.uint16)
        zero_patch_times = np.stack(zero_patch_times, axis=0)
    else:
        zero_patch_coords = np.zeros((0,2), dtype=np.uint16)
        zero_patch_times = np.zeros((0,), dtype=np.int64)
    patch_data = np.stack(patch_data, axis=0)
    patch_coords = np.stack(patch_coords, axis=0).astype(np.uint16)
    patch_times = np.stack(patch_times, axis=0)

    return (patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times)


def save_patches(patch_data, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times, out_fn, zero_value=0, scale=None):

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_patch = ds.createDimension("dim_patch", patch_data.shape[0])
        dim_zero_patch = ds.createDimension("dim_zero_patch", zero_patch_coords.shape[0])
        dim_coord = ds.createDimension("dim_coord", 2)
        dim_height = ds.createDimension("dim_height", patch_data.shape[1])
        dim_width = ds.createDimension("dim_width", patch_data.shape[2])

        var_args = {"zlib": True, "complevel": 1}

        chunksizes = (min(2**10, patch_data.shape[0]), patch_data.shape[1], patch_data.shape[2])
        var_patch = ds.createVariable("patches", patch_data.dtype,
            ("dim_patch","dim_height","dim_width"), chunksizes=chunksizes, **var_args)
        var_patch[:] = patch_data
        
        var_patch_coord = ds.createVariable("patch_coords", patch_coords.dtype,
            ("dim_patch","dim_coord"), **var_args)
        var_patch_coord[:] = patch_coords

        var_patch_time = ds.createVariable("patch_times", patch_times.dtype,
            ("dim_patch",), **var_args)
        var_patch_time[:] = patch_times

        var_zero_patch_coord = ds.createVariable("zero_patch_coords", zero_patch_coords.dtype,
            ("dim_zero_patch","dim_coord"), **var_args)
        var_zero_patch_coord[:] = zero_patch_coords

        var_zero_patch_time = ds.createVariable("zero_patch_times", zero_patch_times.dtype,
            ("dim_zero_patch",), **var_args)
        var_zero_patch_time[:] = zero_patch_times

        ds.zero_value = zero_value

        if scale is not None:
            dim_scale = ds.createDimension("dim_scale", len(scale))
            var_scale = ds.createVariable("scale", scale.dtype, ("dim_scale",), **var_args)
            var_scale[:] = scale


def load_patches(fn, in_memory=True):
    if in_memory:
        with open(fn, 'rb') as f:
            ds_raw = f.read()
        fn = None
    else:
        ds_raw = None

    with netCDF4.Dataset(fn, 'r', memory=ds_raw) as ds:
        patch_data = {       
            "patches": np.array(ds["patches"]),
            "patch_coords": np.array(ds["patch_coords"]),
            "patch_times": np.array(ds["patch_times"]),
            "zero_patch_coords": np.array(ds["zero_patch_coords"]),
            "zero_patch_times": np.array(ds["zero_patch_times"]),
            "zero_value": ds.zero_value
        }
        if "scale" in ds.variables:
            patch_data["scale"] = np.array(ds["scale"])

    return patch_data


def load_all_patches(patch_dir, var):
    files = os.listdir(patch_dir)
    jobs = []
    for fn in files:
        file_var = fn.split("_")[1]        
        if file_var == var:
            fn = os.path.join(patch_dir, fn)
            jobs.append(dask.delayed(load_patches)(fn))
    
    file_data = dask.compute(jobs, scheduler="processes")[0]
    patch_data = {}
    keys = ["patches", "patch_coords", "patch_times",
        "zero_patch_coords", "zero_patch_times"]
    for k in keys:
        patch_data[k] = np.concatenate(
            [fd[k] for fd in file_data],
            axis=0
        )
    patch_data["zero_value"] = file_data[0]["zero_value"]
    if "scale" in file_data[0]:
        patch_data["scale"] = file_data[0]["scale"]

    return patch_data


def unpack_patches(patch_data):
    return (
        patch_data["patches"],
        patch_data["patch_coords"],
        patch_data["patch_times"],
        patch_data["zero_patch_coords"],
        patch_data["zero_patch_times"]
    )
