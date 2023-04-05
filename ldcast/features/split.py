from bisect import bisect_left

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import batch


def get_chunks(
    primary_raw, valid_frac=0.1, test_frac=0.1,
    chunk_seconds=2*24*60*60, random_seed=None
):
    t0 = min(
        primary_raw["patch_times"][0],
        primary_raw["zero_patch_times"][0]
    )
    t1 = max(
        primary_raw["patch_times"][-1],
        primary_raw["zero_patch_times"][-1]
    )+1

    rng = np.random.RandomState(seed=random_seed)
    chunk_limits = np.arange(t0,t1,chunk_seconds)
    num_chunks = len(chunk_limits)-1
    
    chunk_ind = np.arange(num_chunks)
    rng.shuffle(chunk_ind)
    i_valid = int(round(num_chunks * valid_frac))
    i_test = i_valid + int(round(num_chunks * test_frac))
    chunk_ind = {
        "valid": chunk_ind[:i_valid],
        "test": chunk_ind[i_valid:i_test],
        "train": chunk_ind[i_test:]
    }
    def get_chunk_limits(chunk_ind_split):
        return sorted(
            (chunk_limits[i], chunk_limits[i+1])
            for i in chunk_ind_split
        )
    chunks = {
        split: get_chunk_limits(chunk_ind_split)
        for (split, chunk_ind_split) in chunk_ind.items()
    }
    return chunks


def train_valid_test_split(
    raw_data, primary_raw_var, chunks=None, **kwargs
):
    if chunks is None:
        primary = raw_data[primary_raw_var] 
        chunks = get_chunks(primary, **kwargs)

    def split_chunks_from_array(x, chunks_split, times):
        n = 0
        chunk_ind = []
        for (t0,t1) in chunks_split:
            k0 = bisect_left(times, t0)
            k1 = bisect_left(times, t1)
            n += k1 - k0
            chunk_ind.append((k0,k1))
        
        shape = (n,) + x.shape[1:]
        x_chunk = np.empty_like(x, shape=shape)
        
        j0 = 0
        for (k0,k1) in chunk_ind:
            j1 = j0 + (k1-k0)
            x_chunk[j0:j1,...] = x[k0:k1,...]
            j0 = j1

        return x_chunk

    split_raw_data = {
        split: {var: {} for var in raw_data}
        for split in chunks
    }
    for (var, raw_data_var) in raw_data.items():
        for (split, chunks_split) in chunks.items():
            
            split_raw_data[split][var]["patches"] = \
                split_chunks_from_array(
                    raw_data_var["patches"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["patch_coords"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["patch_times"], chunks_split,
                    raw_data_var["patch_times"]
                )
            split_raw_data[split][var]["zero_patch_coords"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_coords"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )
            split_raw_data[split][var]["zero_patch_times"] = \
                split_chunks_from_array(
                    raw_data_var["zero_patch_times"], chunks_split,
                    raw_data_var["zero_patch_times"]
                )

            added_keys = set(split_raw_data[split][var].keys())
            missing_keys = set(raw_data[var].keys()) - added_keys
            for k in missing_keys:
                split_raw_data[split][var][k] = raw_data[var][k]

    return (split_raw_data, chunks)


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        variables, raw, predictors, target, primary_var,
        sampling_bins, sampler_file,
        batch_size=64,
        train_epoch_size=1000, valid_epoch_size=200, test_epoch_size=1000,
        valid_seed=None, test_seed=None,
        **kwargs
    ):
        super().__init__()
        self.batch_gen = {
            split: batch.BatchGenerator(
                variables, raw_var, predictors, target, primary_var,
                sampling_bins=sampling_bins, batch_size=batch_size,
                sampler_file=sampler_file.get(split),
                augment=(split=="train"),
                **kwargs
            )
            for (split,raw_var) in raw.items()
        }
        self.datasets = {}
        if "train" in self.batch_gen:
            self.datasets["train"] = batch.StreamBatchDataset(
                self.batch_gen["train"], train_epoch_size
            )
        if "valid" in self.batch_gen:
            self.datasets["valid"] = batch.DeterministicBatchDataset(
                self.batch_gen["valid"], valid_epoch_size, random_seed=valid_seed
            )
        if "test" in self.batch_gen:
             self.datasets["test"] = batch.DeterministicBatchDataset(
                self.batch_gen["test"], test_epoch_size, random_seed=test_seed
            )

    def dataloader(self, split):
        return DataLoader(
            self.datasets[split], batch_size=None,
            pin_memory=True, num_workers=0
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("valid")

    def test_dataloader(self):
        return self.dataloader("test")
