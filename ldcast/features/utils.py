from numba import njit, prange
import numpy as np
from scipy.signal import convolve


def log_scale_with_zero(range, n=65536, dtype=np.float32):
    scale = np.linspace(np.log10(range[0]), np.log10(range[1]), n-1)
    scale = np.hstack((0, 10**scale)).astype(dtype)
    return scale


def log_quantize_with_zero(x, range, n=65536, dtype=np.uint16):
    scale = log_scale_with_zero(range, n=n, dtype=x.dtype)
    y = np.empty_like(x, dtype=dtype)
    log_quant_with_zeros(x, y, np.log10(scale[1:]))
    return (y, scale)


# optimized helper function for the above
@njit(parallel=True)
def log_quant_with_zeros(x, y, scale):
    x = x.ravel()
    y = y.ravel()
    min_val = 10**scale[0]
    
    for i in prange(x.shape[0]):
        # map small values to 0
        if x[i] < min_val:
            y[i] = 0
            continue
        
        lx = np.log10(x[i])
        if lx >= scale[-1]:
            # map too big values to max of scale
            y[i] = len(scale)
        else:
            # binary search for the rest
            k0 = 0
            k1 = len(scale)
            while k1-k0 > 1:
                km = k0 + (k1-k0)//2
                if lx < scale[km]:
                    k1 = km
                else:
                    k0 = km
            
            if k0 == len(scale)-1:
                q = k0
            elif k0 == 0:
                q = 0
            else:
                d0 = abs(lx-scale[k0])
                d1 = abs(lx-scale[k1])
                if d0 < d1:
                    q = k0
                else:
                    q = k1

            y[i] = q+1 # add 1 to leave space for zero


@njit(parallel=True)
def average_pool(x, factor=2, missing=65535):
    y = np.empty((x.shape[0]//factor, x.shape[1]//factor), dtype=x.dtype)
    N = factor**2
    N_thresh = N//2

    for iy in prange(y.shape[0]):
        ix0 = iy * factor
        ix1 = ix0 + factor
        for jy in range(y.shape[1]):            
            jx0 = jy * factor
            jx1 = jx0 + factor
            v = float(0.0)
            num_valid = 0

            for ix in range(ix0, ix1):
                for jx in range(jx0, jx1):
                    if x[ix,jx] != missing:
                        v += x[ix,jx]
                        num_valid += 1
            
            if num_valid >= N_thresh:
                y[iy,jy] = v/num_valid
            else:
                y[iy,jy] = missing
        
    return y


@njit(parallel=True)
def mode_pool(x, num_values=256, factor=2):
    y = np.empty((x.shape[0]//factor, x.shape[1]//factor), dtype=x.dtype)
    
    for iy in prange(y.shape[0]):
        v = np.empty(num_values, dtype=np.int64)
        ix0 = iy * factor
        ix1 = ix0 + factor
        for jy in range(y.shape[1]):            
            jx0 = jy * factor
            jx1 = jx0 + factor
            v[:] = 0

            for ix in range(ix0, ix1):
                for jx in range(jx0, jx1):
                    v[x[ix,jx]] += 1
            
            y[iy,jy] = v.argmax()
        
    return y


def fill_holes(missing=65535, rad=1):
    def fill(x):
        # identify mask of points to fill
        o = np.ones((2*rad+1,2*rad+1), dtype=np.uint16)
        n = np.prod(o.shape)
        valid = (x != missing)
        num_valid_neighbors = convolve(valid, o, mode='same', method='direct')
        mask = ~valid & (num_valid_neighbors > 0)

        # compute mean of valid points around each fillable point
        fx = x.copy()
        fx[~valid] = 0
        mx = convolve(fx, o.astype(np.float64), mode='same', method='direct')        
        mx = mx[mask] / num_valid_neighbors[mask]
        if np.issubdtype(x.dtype, np.integer):
            mx = mx.round().astype(x.dtype)        
        
        # fill holes with mean
        fx = x.copy()
        fx[mask] = mx
        return fx

    return fill

