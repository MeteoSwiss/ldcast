import contextlib
import gc

import numpy as np
import torch

from .features.transform import Antialiasing
from .models.autoenc import autoenc, encoder
from .models.genforecast import analysis, unet
from .models.diffusion import diffusion, plms


class Forecast:
    def __init__(
        self,
        ldm_weights_fn,
        autoenc_weights_fn,
        past_timesteps=4,
        future_timesteps=20,
        autoenc_time_ratio=4,
        autoenc_hidden_dim=32,
        use_gpu=True,
        verbose=True,
        R_min_value=0.1,
        R_zero_value=0.02,
        R_min_output=0.1,
        R_max_output=118.428,
        log_R_mean=-0.051,
        log_R_std=0.528,
    ):
        self.verbose = verbose
        self.R_min_value = R_min_value
        self.R_zero_value = R_zero_value
        self.R_min_output = R_min_output
        self.R_max_output = R_max_output
        self.log_R_mean = log_R_mean
        self.log_R_std = log_R_std
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.autoenc_time_ratio = autoenc_time_ratio
        self.autoenc_hidden_dim = autoenc_hidden_dim
        self.antialiasing = Antialiasing()
        
        # setup autoencoder        
        enc = encoder.SimpleConvEncoder()
        dec = encoder.SimpleConvDecoder()    
        autoencoder_obs = autoenc.AutoencoderKL(enc, dec)
        autoencoder_obs.load_state_dict(torch.load(autoenc_weights_fn))

        autoencoders = [autoencoder_obs]
        input_patches = [past_timesteps//autoenc_time_ratio]
        input_size_ratios = [1]
        embed_dim = [128]
        analysis_depth = [4]

        # setup forecaster
        analysis_net = analysis.AFNONowcastNetCascade(
            autoencoders,
            input_patches=input_patches,
            input_size_ratios=input_size_ratios,
            train_autoenc=False,
            output_patches=future_timesteps//autoenc_time_ratio,
            cascade_depth=3,
            embed_dim=embed_dim,
            analysis_depth=analysis_depth
        )

        # setup denoiser
        model = unet.UNetModel(in_channels=autoencoder_obs.hidden_width,
            model_channels=256, out_channels=autoencoder_obs.hidden_width,
            num_res_blocks=2, attention_resolutions=(1,2), 
            dims=3, channel_mult=(1, 2, 4), num_heads=8,
            num_timesteps=future_timesteps//autoenc_time_ratio,
            context_ch=analysis_net.cascade_dims
        )

        # create LDM
        self.ldm = diffusion.LatentDiffusion(model, autoencoder_obs, 
            context_encoder=analysis_net)
        self.ldm.load_state_dict(torch.load(ldm_weights_fn))
        
        num_gpus = torch.cuda.device_count()
        if use_gpu and (num_gpus > 0):
            self.ldm.to(device="cuda")

        # setup sampler
        self.sampler = plms.PLMSSampler(self.ldm)
        
        gc.collect()
    
    def __call__(
        self,
        R_past,
        ensemble_members=1,
        batch_size=None,
        num_diffusion_iters=50
    ):
        if batch_size is None:
            batch_size = ensemble_members
        
        if batch_size < ensemble_members:
            batches = []
            remaining_members = ensemble_members
            while remaining_members > 0:
                batches.append(self(
                    R_past, 
                    ensemble_members=min(remaining_members, batch_size),
                    num_diffusion_iters=num_diffusion_iters
                ))
                remaining_members -= batch_size
            return np.concatenate(batches, axis=0)

        x = self.transform_precip(R_past)
        expansion = (batch_size,) + (-1,) * (x.ndim-1)
        x = x.expand(expansion)
        timesteps = self.input_timesteps(x)        
        future_patches = self.future_timesteps // self.autoenc_time_ratio
        gen_shape = (self.autoenc_hidden_dim, future_patches) + \
            (x.shape[-2]//4, x.shape[-1]//4)
        x = [[x, timesteps]]

        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.sampler.sample(
                num_diffusion_iters, 
                batch_size,
                gen_shape,
                x,
                progbar=self.verbose
            )
        y_pred = self.ldm.autoencoder.decode(s)
        R_future = self.inv_transform_precip(y_pred)

        return R_future

    def transform_precip(self, R):
        x = R.copy()
        x[~(x >= self.R_min_value)] = self.R_zero_value
        x = np.log10(x)
        x -= self.log_R_mean
        x /= self.log_R_std
        x = x.reshape((1,)+x.shape)
        x = self.antialiasing(x)
        x = x.reshape((1,)+x.shape)
        return torch.Tensor(x).to(device=self.ldm.device)

    def inv_transform_precip(self, x):        
        x *= self.log_R_std
        x += self.log_R_mean
        R = torch.pow(10, x)
        if self.R_min_output:        
            R[R < self.R_min_output] = 0.0
        if self.R_max_output is not None:
            R[R > self.R_max_output] = self.R_max_output
        R = R[:,0,...]
        return R.to(device='cpu').numpy()

    def input_timesteps(self, x):
        batch_size = x.shape[0]
        t0 = -x.shape[2]+1
        t1 = 1
        timesteps = torch.arange(t0, t1,
            dtype=x.dtype, device=self.ldm.device)
        return timesteps.unsqueeze(0).expand(batch_size,-1)
