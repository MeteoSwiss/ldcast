from bisect import bisect_left
import multiprocessing

import dask
from numba import njit, prange, types
from numba.typed import Dict
import numpy as np

from .patches import unpack_patches


class EqualFrequencySampler:
    def __init__(
        self, bins, patch_data, patch_index, 
        sample_shape, time_range_valid, time_range_sampling=None,        
        timestep_secs=5*60,
        random_seed=None, preselected_samples=None
    ):
        binned_patches = bin_classify_patches_parallel(
            bins,
            *unpack_patches(patch_data),
            zero_value=patch_data.get("zero_value", 0),
            scale=patch_data.get("scale")
        )
        complete_ind = indices_with_complete_sample(
            patch_index, sample_shape, time_range_valid, timestep_secs
        )
        if time_range_sampling is None:
            time_range_sampling = time_range_valid
        self.starting_ind = [
            starting_indices_for_centers(
                p, complete_ind, sample_shape, time_range_sampling, timestep_secs
            )
            for p in binned_patches
        ]
        self.num_bins = len(self.starting_ind)
        self.rng = np.random.RandomState(seed=random_seed)
        self.preselected_samples = preselected_samples
        self.current_ind = np.array([len(ind) for ind in self.starting_ind])

    def get_bin_sample(self, bin_ind):
        patches = self.starting_ind[bin_ind]
        sample_ind = self.current_ind[bin_ind]
        if sample_ind >= patches.shape[0]:
            self.rng.shuffle(patches)
            sample_ind = self.current_ind[bin_ind] = 0
        else:
            self.current_ind[bin_ind] += 1
        return patches[sample_ind,:]

    def __call__(self, num):
        # sample each bin with equal probability
        bins = self.rng.randint(self.num_bins, size=num)
        coords = np.stack(
            [self.get_bin_sample(b) for b in bins],
            axis=0
        )
        return coords


def bin_classify_patches(
    bins, patches, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times,
    zero_value=0, metric_func=None,
    scale=None,
):
    if metric_func is None:
        def metric_func(x):
            xm = np.percentile(x, 99, axis=(1,2))
            if np.issubdtype(x.dtype, np.integer):
                xm = xm.round()
            return xm.astype(x.dtype)
    
    binned_patches = [[] for _ in range(len(bins)+1)]

    def find_bin(value):
        return bisect_left(bins, value)

    zero_bin = find_bin(zero_value if scale is None else scale[zero_value])
    for (t,(pi,pj)) in zip(zero_patch_times, zero_patch_coords):
        binned_patches[zero_bin].append((t,pi,pj))

    patch_metrics = metric_func(patches)
    if scale is not None:
        patch_metrics = scale[patch_metrics]
    for (metric,t,(pi,pj)) in zip(patch_metrics, patch_times, patch_coords):
        patch_bin = find_bin(metric)
        binned_patches[patch_bin].append((t,pi,pj))

    for i in range(len(binned_patches)):
        if binned_patches[i]:
            binned_patches[i] = np.array(binned_patches[i])
        else:
            binned_patches[i] = np.zeros((0,3), dtype=np.int64)

    return binned_patches


def bin_classify_patches_parallel(
    bins, patches, patch_coords, patch_times,
    zero_patch_coords, zero_patch_times,
    zero_value=0, metric_func=None,
    scale=None,
):
    num_patches = patches.shape[0]
    num_zeros = zero_patch_coords.shape[0]
    num_procs = multiprocessing.cpu_count()
    
    tasks = []
    for p in range(num_procs):
        pk0 = int(round(num_patches*p/num_procs))
        pk1 = int(round(num_patches*(p+1)/num_procs))
        zk0 = int(round(num_zeros*p/num_procs))
        zk1 = int(round(num_zeros*(p+1)/num_procs))
        
        task = dask.delayed(bin_classify_patches)(
            bins,
            patches[pk0:pk1,...], patch_coords[pk0:pk1,...],
            patch_times[pk0:pk1],
            zero_patch_coords[zk0:zk1,...], zero_patch_times[zk0:zk1],
            zero_value=zero_value, metric_func=metric_func,
            scale=scale
        )
        tasks.append(task)
    
    chunked_bins = dask.compute(tasks, scheduler="threads")[0]

    n_bins = len(chunked_bins[0])
    binned_patches = [
        np.concatenate([cb[i] for cb in chunked_bins], axis=0)
        for i in range(n_bins)
    ]
    return binned_patches


def indices_with_complete_sample(
    patch_index, sample_shape, time_range, timestep_secs
):
    """Check which locations will give a sample without missing data.
    """
    ind = np.array(list(patch_index.patch_index.keys()))
    t0 = ind[:,0]
    i0 = ind[:,1]
    j0 = ind[:,2]
    n = ind.shape[0]
    complete = np.ones(n, dtype=bool)
    # we use this dict like a set - numba doesn't support typed sets
    complete_ind = Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.uint8
    )

    @njit(parallel=True) # many nested loops, numba optimization needed
    def check_complete(index, complete, complete_ind):        
        for k in prange(n):
            for ts in range(*time_range):
                t = t0[k] + ts*timestep_secs
                for di in range(sample_shape[0]):
                    i = i0[k] + di
                    for dj in range(sample_shape[1]):
                        j = j0[k] + dj
                        if (t,i,j) not in index:
                            complete[k] = False
        
        for k in range(n): # no prange: can't set dict items in parallel
            if complete[k]:
                complete_ind[(t0[k],i0[k],j0[k])] = np.uint8(0)
    
    check_complete(patch_index.patch_index, complete, complete_ind)

    return complete_ind


def starting_indices_for_centers(
    centers, complete_ind, sample_shape, time_range, timestep_secs
):
    """Determine a complete list of sample indices that
    contain one or more of the centerpoints.
    """

    @njit
    def find_indices(centers, starting_ind, complete_ind):
        for k in range(centers.shape[0]):
            t0 = centers[k,0]
            i0 = centers[k,1]
            j0 = centers[k,2]
            for ts in range(*time_range):
                t = t0 - ts*timestep_secs # note minus signs in (t,i,j)
                for di in range(sample_shape[0]):
                    i = i0 - di
                    for dj in range(sample_shape[1]):
                        j = j0 - dj
                        if (t,i,j) in complete_ind:
                            starting_ind[(t,i,j)] = np.uint8(0)

    num_chunks = multiprocessing.cpu_count()
    @dask.delayed
    def chunk(i):
        starting_ind = Dict.empty(
            key_type=types.UniTuple(types.int64, 3),
            value_type=types.uint8
        )
        k0 = int(round(centers.shape[0] * (i / num_chunks)))
        k1 = int(round(centers.shape[0] * ((i+1) / num_chunks)))
        find_indices(centers[k0:k1,...], starting_ind, complete_ind)
        return starting_ind

    jobs = [chunk(i) for i in range(num_chunks)]
    starting_ind = dask.compute(jobs, scheduler='threads')[0]
    starting_ind = np.concatenate(
        [np.array(list(st_ind.keys())) for st_ind in starting_ind if st_ind],
        axis=0
    )
    return starting_ind
