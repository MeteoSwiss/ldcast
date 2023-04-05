# following https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html
from datetime import timedelta

import dask
import numpy as np
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation


class PySTEPSModel:
    def __init__(
        self,
        data_format='channels_first',
        future_timesteps=20,
        ensemble_size=32,
        km_per_pixel=1.0,
        interval=timedelta(minutes=5),
        transform_to_rainrate=None,
        transform_from_rainrate=None,
    ):
        self.transform_to_rainrate = transform_to_rainrate
        self.transform_from_rainrate = transform_from_rainrate
        self.data_format = data_format
        self.nowcast_method = nowcasts.get_method("steps")
        self.future_timesteps = future_timesteps
        self.ensemble_size = ensemble_size
        self.km_per_pixel = km_per_pixel
        self.interval = interval

    def zero_prediction(self, R, zerovalue):
        out_shape = (self.future_timesteps,) + R.shape[1:] + \
            (self.ensemble_size,)
        return np.full(out_shape, zerovalue, dtype=R.dtype)

    def predict_sample(self, x, threshold=-10.0, zerovalue=-15.0):
        R = self.transform_to_rainrate(x)        
        (R, _) = transformation.dB_transform(
            R, threshold=0.1, zerovalue=zerovalue
        )
        R[~np.isfinite(R)] = zerovalue
        if (R == zerovalue).all():
            R_f = self.zero_prediction(R, zerovalue)
        else:
            V = dense_lucaskanade(R)
            try:
                R_f = self.nowcast_method(
                    R,
                    V,
                    self.future_timesteps,
                    n_ens_members=self.ensemble_size,
                    n_cascade_levels=6,
                    precip_thr=threshold,
                    kmperpixel=self.km_per_pixel,
                    timestep=self.interval.total_seconds()/60,
                    noise_method="nonparametric",
                    vel_pert_method="bps",
                    mask_method="incremental",
                    num_workers=2
                )
                R_f = R_f.transpose(1,2,3,0)
            except (ValueError, RuntimeError) as e:
                zero_error = str(e).endswith("contains non-finite values") or \
                    str(e).startswith("zero-size array to reduction operation") or \
                    str(e).endswith("nonstationary AR(p) process")
                if zero_error:
                    # occasional PySTEPS errors that happen with little/no precip
                    # therefore returning all zeros makes sense
                    R_f = self.zero_prediction(R, zerovalue)
                else:
                    raise

        # Back-transform to rain rates
        R_f = transformation.dB_transform(
            R_f, threshold=threshold, inverse=True
        )[0]

        if self.transform_from_rainrate is not None:
            R_f = self.transform_from_rainrate(R_f)

        return R_f

    def __call__(self, x, parallel=True):        
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = np.array(x, copy=False)
        if self.data_format == "channels_first":
            x = x.transpose(0,2,3,4,1)

        pred = self.predict_sample
        if parallel:
            pred = dask.delayed(pred)
        y = [
            pred(x[i,:,:,:,0]) 
            for i in range(x.shape[0])    
        ]
        if parallel:
            y = dask.compute(y, scheduler="threads", num_workers=len(y))[0]
        y = np.stack(y, axis=0)
        
        if self.data_format == "channels_first":
            y = np.expand_dims(y, 1)
        else:
            y = np.expand_dims(y, -2)

        return y
