import concurrent.futures
import multiprocessing

from numba import njit, prange
import numpy as np
from scipy.ndimage import convolve


def quick_cast(x, y):        
    num_threads = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        futures = {}
        limits = np.linspace(0, x.shape[0], num_threads+1).round().astype(int)
        def _cast(k0,k1):
            y[k0:k1,...] = x[k0:k1,...]        
        for k in range(len(limits)-1):
            args = (_cast, limits[k], limits[k+1])
            futures[executor.submit(*args)] = k
        concurrent.futures.wait(futures)


def cast(dtype=np.float16):
    xc = None
    def transform(raw):
        nonlocal xc
        if (xc is None) or (xc.shape != raw.shape):
            xc = np.empty_like(raw, dtype=dtype)
        quick_cast(raw, xc)
        return xc
    return transform


@njit(parallel=True)
def scale_array(in_arr, out_arr, scale):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = scale[in_arr[i]]

# NumPy version
#def scale_array(in_arr, out_arr, scale):
#    out_arr[:] = scale[in_arr]

def normalize(mean=0.0, std=1.0, dtype=np.float32):
    scaled = scaled_dt = None

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        normalize_array(raw, scaled, mean, std)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


def normalize_threshold(mean=0.0, std=1.0, threshold=0.0, fill_value=0.0, log=False):
    scaled = None

    def transform(raw):
        nonlocal scaled
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)        
        normalize_threshold_array(raw, scaled, mean, std, threshold, fill_value, log=log)

        return scaled

    return transform


def scale_log_norm(scale, threshold=None, missing_value=None,
    fill_value=0, mean=0.0, std=1.0, dtype=np.float32):

    log_scale = np.log10(scale).astype(np.float32)
    if threshold is not None:
        log_scale[log_scale < np.log10(threshold)] = np.log10(fill_value)
    if missing_value is not None:
        log_scale[missing_value] = np.log10(fill_value)
    log_scale[~np.isfinite(log_scale)] = np.log10(fill_value)
    log_scale -= mean
    log_scale /= std
    scaled = scaled_dt = None

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        scale_array(raw, scaled, log_scale)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


def combine(transforms, memory_format="channels_first", dim=3):
    #combined = None    
    channels_axis = 1 if (memory_format == "channels_first") else -1

    def transform(*raw):
        #nonlocal combined
        transformed = [t(r) for (t, r) in zip(transforms, raw)]
        for i in range(len(transformed)):
            if transformed[i].ndim == dim + 1:
                transformed[i] = np.expand_dims(transformed[i], channels_axis)

        return np.concatenate(transformed, axis=channels_axis)

    return transform


class Antialiasing:
    def __init__(self):
        (x,y) = np.mgrid[-2:3,-2:3]
        self.kernel = np.exp(-0.5*(x**2+y**2)/(0.5**2))
        self.kernel /= self.kernel.sum()
        self.edge_factors = {}
        self.img_smooth = {}
        num_threads = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(num_threads)

    def __call__(self, img):
        img_shape = img.shape[-2:]
        if img_shape not in self.edge_factors:
            s = convolve(np.ones(img_shape, dtype=np.float32),
                self.kernel, mode="constant")
            s = 1.0/s
            self.edge_factors[img_shape] = s
        else:
            s = self.edge_factors[img_shape]
        
        if img.shape not in self.img_smooth:
            img_smooth = np.empty_like(img)
            self.img_smooth[img_shape] = img_smooth
        else:
            img_smooth = self.img_smooth[img_shape]

        def _convolve_frame(i,j):
            convolve(img[i,j,:,:], self.kernel, 
                mode="constant", output=img_smooth[i,j,:,:])
            img_smooth[i,j,:,:] *= s

        futures = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                args = (_convolve_frame, i, j)
                futures.append(self.executor.submit(*args))
        concurrent.futures.wait(futures)

        return img_smooth


def default_rainrate_transform(scale):
    scaling = scale_log_norm(
        scale, threshold=0.1, fill_value=0.02,
        mean=-0.051, std=0.528, dtype=np.float32
    )
    antialiasing = Antialiasing()
    def transform(raw):
        x = scaling(raw)
        return antialiasing(x)
    return transform


def scale_norm(scale, threshold=None, missing_value=None,
    fill_value=0, mean=0.0, std=1.0, dtype=np.float32):

    scale = scale.astype(np.float32).copy()
    scale[np.isnan(scale)] = fill_value
    if threshold is not None:
        scale[scale < threshold] = fill_value
    if missing_value is not None:
        missing_value = np.atleast_1d(missing_value)
        for m in missing_value:
            scale[m] = fill_value
    scale -= mean
    scale /= std
    scaled = scaled_dt = None    

    def transform(raw):
        nonlocal scaled, scaled_dt
        if (scaled is None) or (scaled.shape != raw.shape):
            scaled = np.empty_like(raw, dtype=np.float32)
            scaled_dt = np.empty_like(raw, dtype=dtype)
        scale_array(raw, scaled, scale)

        if dtype == np.float32:
            return scaled
        else:
            quick_cast(scaled, scaled_dt)
            return scaled_dt

    return transform


@njit(parallel=True)
def threshold_array(in_arr, out_arr, threshold):
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = np.float32(in_arr[i] >= threshold)


def one_hot(values):    
    translation = np.zeros(max(values)+1, dtype=int)
    num_categories = len(values)
    for (i,v) in enumerate(values):
        translation[v] = i
    onehot = onehot_dt = None

    def transform(raw):
        nonlocal onehot, onehot_dt
        if (onehot is None) or (onehot.shape[:-1] != raw.shape):
            onehot = np.empty(raw.shape+(num_categories,),
                dtype=np.float32)
            onehot = np.empty(raw.shape+(num_categories,),
                dtype=np.uint8)
        onehot_transform(raw, onehot, translation)
        quick_cast(onehot, onehot_dt)

        return onehot

    return transform
            
    
@njit(parallel=True)
def onehot_transform(in_arr, out_arr, translation):
    for k in prange(in_arr.shape[0]):
        out_arr[k,...] = 0.0
        for t in range(in_arr.shape[1]):
            for i in range(in_arr.shape[2]):
                for j in range(in_arr.shape[3]):
                    ind = np.uint64(in_arr[k,t,i,j])
                    c = translation[ind]
                    out_arr[k,t,i,j,c] = 1.0


@njit(parallel=True)
def normalize_array(in_arr, out_arr, mean, std):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        out_arr[i] = (in_arr[i]-mean)*inv_std


@njit(parallel=True)
def normalize_threshold_array(
    in_arr, out_arr, 
    mean, std, 
    threshold, fill_value, log=False
):
    mean = np.float32(mean)
    inv_std = np.float32(1.0/std)
    threshold = np.float32(threshold)
    fill_value = np.float32(fill_value)
    in_arr = in_arr.ravel()
    out_arr = out_arr.ravel()
    for i in prange(in_arr.shape[0]):
        x = in_arr[i]
        if x < threshold:
            x = fill_value
        if log:
            x = np.log10(x)
        out_arr[i] = (x-mean)*inv_std


# NumPy version
#def threshold_array(in_arr, out_arr, threshold):
#    out_arr[:] = (in_arr >= threshold).astype(np.float32)


def R_threshold(scale, threshold):    
    thresholded = None
    scale_treshold = np.nanargmax(scale > threshold)

    def transform(rzc_raw):
        nonlocal thresholded
        if (thresholded is None) or (thresholded.shape != rzc_raw.shape):
            thresholded = np.empty_like(rzc_raw, dtype=np.float32)
        threshold_array(rzc_raw, thresholded, scale_treshold)

        return thresholded

    return transform
