from datetime import datetime, timedelta
import os
import pickle

from numba import njit, prange, types
from numba.typed import Dict
import numpy as np
from torch.utils.data import Dataset, IterableDataset

from .patches import unpack_patches
from .sampling import EqualFrequencySampler


class BatchGenerator:
    def __init__(self, 
        variables,
        raw,
        predictors,
        target,
        primary_var,
        time_range_sampling=(-1,2),
        forecast_raw_vars=(),
        sampling_bins=None,
        sampler_file=None,
        sample_shape=(4,4),
        batch_size=32,
        interval=timedelta(minutes=5),
        random_seed=None,
        augment=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.interval = interval
        self.interval_secs = np.int64(self.interval.total_seconds())        
        self.variables = variables
        self.predictors = predictors
        self.target = target
        self.used_variables = predictors + [target]
        self.rng = np.random.RandomState(seed=random_seed)
        self.augment = augment

        # setup indices for retrieving source raw data
        self.sources = set.union(
            *(set(variables[v]["sources"]) for v in self.used_variables)
        )
        self.forecast_raw_vars = set(forecast_raw_vars) & self.sources
        self.patch_index = {}
        for raw_name_base in self.sources:
            if raw_name_base in forecast_raw_vars:
                raw_names = (
                    rn for rn in raw if rn.startswith(raw_name_base+"-")
                )
            else:
                raw_names = (raw_name_base,)
            for raw_name in raw_names:
                raw_data = raw[raw_name]
                self.setup_index(raw_name, raw_data, sample_shape)
   
        for raw_name in self.forecast_raw_vars:
            patch_index_var = {
                k: v for (k,v) in self.patch_index.items()
                if k.startswith(raw_name+"-")
            }
            self.patch_index[raw_name] = \
                ForecastPatchIndexWrapper(patch_index_var)

        # setup samplers
        if (sampler_file is None) or not os.path.isfile(sampler_file):
            print("No cached sampler found, creating a new one...")
            primary_raw_var = variables[primary_var]["sources"][0]
            t0 = t1 = None
            for (var_name, var_data) in variables.items():
                timesteps = var_data["timesteps"][[0,-1]].copy()
                timesteps[0] -= 1
                ts_secs = timesteps * \
                    var_data.get("timestep_secs", self.interval_secs)
                timesteps = ts_secs // self.interval_secs
                t0 = timesteps[0] if t0 is None else min(t0,timesteps[0])
                t1 = timesteps[-1] if t1 is None else max(t1,timesteps[-1])
            time_range_valid = (t0,t1+1)
            self.sampler = EqualFrequencySampler(
                sampling_bins, raw[primary_raw_var],
                self.patch_index[primary_raw_var], sample_shape,
                time_range_valid, time_range_sampling=time_range_sampling,
                timestep_secs=self.interval_secs
            )
            if sampler_file is not None:                
                print(f"Caching sampler to {sampler_file}.")
                with open(sampler_file, 'wb') as f:
                    pickle.dump(self.sampler, f)
        else:
            print(f"Loading cached sampler from {sampler_file}.")
            with open(sampler_file, 'rb') as f:
                self.sampler = pickle.load(f)

    def setup_index(self, raw_name, raw_data, box_size):
        zero_value = raw_data.get("zero_value", 0)
        missing_value = raw_data.get("missing_value", zero_value)

        self.patch_index[raw_name] = PatchIndex(
            *unpack_patches(raw_data),
            zero_value=zero_value,
            missing_value=missing_value,
            interval=self.interval,
            box_size=box_size
        )
    
    def augmentations(self):
        return tuple(self.rng.randint(2, size=3))
    
    def augment_batch(self, batch, transpose, flipud, fliplr):
        if self.augment:
            if transpose:
                axes = list(range(batch.ndim))
                axes = axes[:-2] + [axes[-1], axes[-2]]
                batch = batch.transpose(axes)
            flips = []
            if flipud:
                flips.append(-2) 
            if fliplr:
                flips.append(-1)
            if flips:
                batch = np.flip(batch, axis=flips)
        return batch.copy()

    def batch(self, samples=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if samples is None:
            # get the sample coordinates from the sampler
            samples = self.sampler(batch_size)
        (t0,i0,j0) = samples.T

        if self.augment:
            augmentations = self.augmentations()

        batch = {}
        for var_name in self.used_variables:
            var_data = self.variables[var_name]  

            # different timestep from standard (e.g. forecast); round down
            # to times where we have data available
            ts_secs = var_data.get("timestep_secs", self.interval_secs)
            t_shift = -(t0 % ts_secs)
            t0_shifted = t0 + t_shift
            t = t0_shifted[:,None] + ts_secs*var_data["timesteps"][None,:]
            t_relative = (t - t0[:,None]) / self.interval_secs
                      
            # read raw data from index
            raw_data = (
                self.patch_index[raw_name](t,i0,j0)
                for raw_name in var_data["sources"]
            )
            
            # transform to model variable
            batch_var = var_data["transform"](*raw_data)
            
            # add channel dimension if not already present
            add_dims = (1,) if batch_var.ndim == 4 else ()
            batch_var = np.expand_dims(batch_var, add_dims)

            # data augmentation
            if self.augment:
                batch_var = self.augment_batch(batch_var, *augmentations)
            
            # bundle with time coordinates
            batch[var_name] = (batch_var, t_relative.astype(np.float32))

        pred_batch = [batch[v] for v in self.predictors]
        target_batch = batch[self.target][0] # no time coordinates for target
        return (pred_batch, target_batch)

    def batches(self, *args, num=None, **kwargs):
        if num is not None:
            for i in range(num):
                yield self.batch(*args, **kwargs)
        else:
            while True:
                yield self.batch(*args, **kwargs)


class StreamBatchDataset(IterableDataset):
    def __init__(self, batch_gen, batches_per_epoch):
        super().__init__()
        self.batch_gen = batch_gen
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        batches = self.batch_gen.batches(num=self.batches_per_epoch)
        yield from batches


class DeterministicBatchDataset(Dataset):
    def __init__(self, batch_gen, batches_per_epoch, random_seed=None):
        super().__init__()
        self.batch_gen = batch_gen
        self.batches_per_epoch = batches_per_epoch
        self.batch_gen.sampler.rng = np.random.RandomState(seed=random_seed)
        self.samples = [
            self.batch_gen.sampler(self.batch_gen.batch_size)
            for i in range(self.batches_per_epoch)
        ]

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, ind):
        return self.batch_gen.batch(samples=self.samples[ind])


class PatchIndex:
    IDX_ZERO = -1
    IDX_MISSING = -2

    def __init__(
        self, patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times,
        interval=timedelta(minutes=5),
        box_size=(4,4), zero_value=0,
        missing_value=0
    ):
        self.dt = int(round(interval.total_seconds()))
        self.box_size = box_size
        self.zero_value = zero_value
        self.missing_value = missing_value
        self.patch_data = patch_data
        self.sample_shape = (
            self.patch_data.shape[1]*box_size[0],
            self.patch_data.shape[2]*box_size[1]
        )

        self.patch_index = Dict.empty(
            key_type=types.UniTuple(types.int64, 3),
            value_type=types.int64
        )
        init_patch_index(self.patch_index, patch_coords, patch_times)
        init_patch_index_zero(self.patch_index, zero_patch_coords,
            zero_patch_times, PatchIndex.IDX_ZERO)

        self._batch = None

    def _alloc_batch(self, batch_size, num_timesteps):
        needs_rebuild = (self._batch is None) or \
            (self._batch.shape[0] < batch_size) or \
            (self._batch.shape[1] < num_timesteps)
        if needs_rebuild:
            del self._batch
            self._batch = np.zeros(
                (batch_size,num_timesteps)+self.sample_shape,
                self.patch_data.dtype
            )
        return self._batch

    def __call__(self, t, i0_all, j0_all):
        batch = self._alloc_batch(*t.shape)

        i1_all = i0_all + self.box_size[0]
        j1_all = j0_all + self.box_size[1]
        bi_size = self.patch_data.shape[1]
        bj_size = self.patch_data.shape[2]

        build_batch(batch, self.patch_data, self.patch_index, 
            t, i0_all, i1_all, j0_all, j1_all,
            bi_size, bj_size, self.zero_value, 
            self.missing_value)

        return batch[:,:t.shape[1],...]


@njit
def init_patch_index(patch_index, patch_coords, patch_times):
    for k in range(patch_coords.shape[0]):
        t = patch_times[k]
        i = np.int64(patch_coords[k,0])
        j = np.int64(patch_coords[k,1])
        patch_index[(t,i,j)] = k


@njit
def init_patch_index_zero(patch_index, zero_patch_coords, 
    zero_patch_times, idx_zero):

    for k in range(zero_patch_coords.shape[0]):
        t = zero_patch_times[k]
        i = np.int64(zero_patch_coords[k,0])
        j = np.int64(zero_patch_coords[k,1])
        patch_index[(t,i,j)] = idx_zero


# numba can't find these values from PatchIndex
IDX_ZERO = PatchIndex.IDX_ZERO
IDX_MISSING = PatchIndex.IDX_MISSING
@njit(parallel=True)
def build_batch(
    batch, patch_data, patch_index,
    t_all, i0_all, i1_all, j0_all, j1_all,
    bi_size, bj_size, zero_value, missing_value
):
    for k in prange(t_all.shape[0]):
        i0 = i0_all[k]
        i1 = i1_all[k]
        j0 = j0_all[k]
        j1 = j1_all[k]

        for (bt,t) in enumerate(t_all[k,:]):
            for i in range(i0, i1):
                bi0 = (i-i0) * bi_size
                bi1 = bi0 + bi_size
                for j in range(j0, j1):
                    ind = int(patch_index.get((t,i,j), IDX_MISSING))
                    bj0 = (j-j0) * bj_size
                    bj1 = bj0 + bj_size
                    if ind >= 0:
                        batch[k,bt,bi0:bi1,bj0:bj1] = patch_data[ind]
                    elif ind == IDX_ZERO:
                        batch[k,bt,bi0:bi1,bj0:bj1] = zero_value
                    elif ind == IDX_MISSING:
                        batch[k,bt,bi0:bi1,bj0:bj1] = missing_value


class ForecastPatchIndexWrapper(PatchIndex):
    def __init__(self, patch_index):        
        self.patch_index = patch_index
        raw_names = {"-".join(v.split("-")[:-1]) for v in patch_index}
        if len(raw_names) != 1:
            raise ValueError(
                "Can only wrap variables with the same base name")
        self.raw_name = list(raw_names)[0]
        lags_hour = [int(v.split("-")[-1]) for v in patch_index]
        self.lags_hour = set(lags_hour)
        forecast_interval_hour = np.diff(sorted(lags_hour))        
        if len(set(forecast_interval_hour)) != 1:
            raise ValueError("Lags must be evenly spaced")        
        forecast_interval_hour = forecast_interval_hour[0]
        if (24 % forecast_interval_hour):
            raise ValueError(
                "24 hours must be a multiple of the forecast interval")
        self.forecast_interval_hour = forecast_interval_hour
        self.forecast_interval = 3600 * forecast_interval_hour
        
        # need to set these for _alloc_batch to work
        self._batch = None
        v = list(self.patch_index.keys())[0]
        self.sample_shape = self.patch_index[v].sample_shape
        self.patch_data = self.patch_index[v].patch_data

    def __call__(self, t, i0, j0):
        batch = self._alloc_batch(*t.shape)

        # ensure that all data come from the same forecast
        t0 = t[:,:1]
        start_time_from_fc = t0 % self.forecast_interval
        time_from_fc = start_time_from_fc + (t - t0)
        lags_hour = (time_from_fc // self.forecast_interval) * \
            self.forecast_interval_hour
        
        for lag in self.lags_hour:
            raw_name_lag = f"{self.raw_name}-{lag}"
            batch_lag = self.patch_index[raw_name_lag](t,i0,j0)
            lag_mask = (lags_hour == lag)
            copy_masked_times(batch_lag, batch, lag_mask)

        return batch[:,:t.shape[1],...]


@njit(parallel=True)
def copy_masked_times(from_batch, to_batch, mask):
    for k in prange(from_batch.shape[0]):
        for bt in range(from_batch.shape[1]):
            if mask[k,bt]:
                to_batch[k,bt,:,:] = from_batch[k,bt,:,:]
