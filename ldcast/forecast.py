import contextlib
import gc

import numpy as np
import torch
import torch.multiprocessing as mp

from .features.transform import Antialiasing
from .models.autoenc import autoenc, encoder
from .models.genforecast import analysis, unet
from .models.diffusion import diffusion, plms


class Forecast:
    def __init__(
        self,
        *,
        ldm_weights_fn,
        autoenc_weights_fn,
        gpu='auto',
        past_timesteps=4,
        future_timesteps=20,
        autoenc_time_ratio=4,
        autoenc_hidden_dim=32,
        verbose=True,
        R_min_value=0.1,
        R_zero_value=0.02,
        R_min_output=0.1,
        R_max_output=118.428,
        log_R_mean=-0.051,
        log_R_std=0.528,
    ):
        self.ldm_weights_fn = ldm_weights_fn
        self.autoenc_weights_fn = autoenc_weights_fn
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

        # setup LDM
        self.ldm = self._init_model()
        if gpu is not None:
            if gpu == 'auto':
                if torch.cuda.device_count() > 0:
                    self.ldm.to(device="cuda")
            else:
                self.ldm.to(device=f"cuda:{gpu}")

        # setup sampler
        self.sampler = plms.PLMSSampler(self.ldm)

        gc.collect()

    def _init_model(self):
        # setup autoencoder        
        enc = encoder.SimpleConvEncoder()
        dec = encoder.SimpleConvDecoder()
        autoencoder_obs = autoenc.AutoencoderKL(enc, dec)
        autoencoder_obs.load_state_dict(torch.load(self.autoenc_weights_fn))

        autoencoders = [autoencoder_obs]
        input_patches = [self.past_timesteps//self.autoenc_time_ratio]
        input_size_ratios = [1]
        embed_dim = [128]
        analysis_depth = [4]

        # setup forecaster
        analysis_net = analysis.AFNONowcastNetCascade(
            autoencoders,
            input_patches=input_patches,
            input_size_ratios=input_size_ratios,
            train_autoenc=False,
            output_patches=self.future_timesteps//self.autoenc_time_ratio,
            cascade_depth=3,
            embed_dim=embed_dim,
            analysis_depth=analysis_depth
        )

        # setup denoiser
        denoiser = unet.UNetModel(in_channels=autoencoder_obs.hidden_width,
            model_channels=256, out_channels=autoencoder_obs.hidden_width,
            num_res_blocks=2, attention_resolutions=(1,2), 
            dims=3, channel_mult=(1, 2, 4), num_heads=8,
            num_timesteps=self.future_timesteps//self.autoenc_time_ratio,
            context_ch=analysis_net.cascade_dims
        )

        # create LDM
        ldm = diffusion.LatentDiffusion(denoiser, autoencoder_obs, 
            context_encoder=analysis_net)
        ldm.load_state_dict(torch.load(self.ldm_weights_fn))

        return ldm
    
    def __call__(
        self,
        R_past,
        num_diffusion_iters=50
    ):
        # preprocess inputs and setup correct input shape
        x = self.transform_precip(R_past)
        timesteps = self.input_timesteps(x)        
        future_patches = self.future_timesteps // self.autoenc_time_ratio
        gen_shape = (self.autoenc_hidden_dim, future_patches) + \
            (x.shape[-2]//4, x.shape[-1]//4)
        x = [[x, timesteps]]

        # run LDM sampler
        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.sampler.sample(
                num_diffusion_iters, 
                x[0][0].shape[0],
                gen_shape,
                x,
                progbar=self.verbose
            )

        # postprocess outputs
        y_pred = self.ldm.autoencoder.decode(s)
        R_pred = self.inv_transform_precip(y_pred)

        return R_pred[0,...]

    def transform_precip(self, R):
        x = R.copy()
        x[~(x >= self.R_min_value)] = self.R_zero_value
        x = np.log10(x)
        x -= self.log_R_mean
        x /= self.log_R_std
        x = x.reshape((1,) + x.shape)
        x = self.antialiasing(x)
        x = x.reshape((1,) + x.shape)
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


class ForecastDistributed:
    def __init__(
        self,
        ldm_weights_fn,
        autoenc_weights_fn,
        past_timesteps=4,
        future_timesteps=20,
        autoenc_time_ratio=4,
        autoenc_hidden_dim=32,
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
       
       # start worker processes
        context = mp.get_context('spawn')
        self.input_queue = context.Queue()
        self.output_queue = context.Queue()
        process_kwargs = {
            "past_timesteps": past_timesteps,
            "future_timesteps": future_timesteps,
            "ldm_weights_fn": ldm_weights_fn,
            "autoenc_weights_fn": autoenc_weights_fn,
            "autoenc_time_ratio": autoenc_time_ratio,
            "autoenc_hidden_dim": autoenc_hidden_dim,
            "R_min_value": R_min_value,
            "R_zero_value": R_zero_value,
            "R_min_output": R_min_output,
            "R_max_output": R_max_output,
            "log_R_mean": log_R_mean,
            "log_R_std": log_R_std,
            "verbose": True
        }
        self.num_procs = max(1, torch.cuda.device_count())
        self.compute_procs = mp.spawn(
            _compute_process,
            args=(self.input_queue, self.output_queue, process_kwargs),
            nprocs=self.num_procs,
            join=False            
        )

        # wait for worker processes to be ready
        for _ in range(self.num_procs):
            self.output_queue.get()

        gc.collect()

    def __call__(
        self,
        R_past,
        ensemble_members=1,
        num_diffusion_iters=50
    ):
        # send samples to compute processes
        for (i, R_past_sample) in enumerate(R_past):
            for j in range(ensemble_members):
                self.input_queue.put((R_past_sample, num_diffusion_iters, i, j))
        
        # build output array
        pred_shape = (R_past.shape[0], self.future_timesteps) + \
            R_past.shape[2:] + (ensemble_members,)
        R_pred = np.empty(pred_shape, R_past.dtype)
        
        # gather outputs from processes
        predictions_needed = R_past.shape[0] * ensemble_members
        for _ in range(predictions_needed):
            (R_pred_sample, i, j) = self.output_queue.get()
            R_pred[i,...,j] = R_pred_sample

        return R_pred

    def __del__(self):
        for _ in range(self.num_procs):
            self.input_queue.put(None)
        self.compute_procs.join()


def _compute_process(process_index, input_queue, output_queue, kwargs):
    gpu = process_index if (torch.cuda.device_count() > 0) else None
    fc = Forecast(gpu=gpu, **kwargs)
    output_queue.put("Ready") # signal process ready to accept inputs

    while (data := input_queue.get()) is not None:
        (R_past, num_diffusion_iters, sample, member) = data
        R_pred = fc(R_past, num_diffusion_iters=num_diffusion_iters)
        output_queue.put((R_pred, sample, member))
