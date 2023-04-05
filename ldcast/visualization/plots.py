import concurrent
import multiprocessing
import netCDF4
import os

from matplotlib import gridspec, colors, pyplot as plt
import numpy as np
import torch

from ..analysis import rank, fss
from ..features import io
from .cm import homeyer_rainbow


def reverse_transform_R(R, mean=-0.051, std=0.528):
    return 10**(R*std + mean)


def plot_precip_image(
    ax, R,     
    Rmin=-3.1212, Rmax=4.168956, threshold_mmh=0.1,
    transform_R=False,
    grid_spacing=64
):
    if isinstance(R, torch.Tensor):
        R = R.detach().numpy()
    if transform_R:
        R = reverse_transform_R(R, mean=mean, std=std)
    Rmin = reverse_transform_R(Rmin)
    Rmax = reverse_transform_R(Rmax)
    if threshold_mmh:
        Rmin = max(Rmin, threshold_mmh)
        R[R < threshold_mmh] = np.nan
    norm = colors.LogNorm(Rmin, Rmax)
    ax.set_yticks(np.arange(0, R.shape[0], grid_spacing))
    ax.set_xticks(np.arange(0, R.shape[1], grid_spacing))
    ax.grid(which='major', alpha=0.35)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

    return ax.imshow(R, norm=norm, cmap=homeyer_rainbow)


def plot_autoencoder_reconstruction(
    R, R_hat, samples=8, timesteps=4,
    out_file=None
):   
    fig = plt.figure(figsize=(2*timesteps*2+0.5, samples*2), dpi=150)
    
    gs = gridspec.GridSpec(
        samples, 2*timesteps+1,
        width_ratios=(1,)*(2*timesteps)+(0.2,),
        wspace=0.02, hspace=0.02
    )
    for k in range(samples):
        for (i,j) in enumerate(range(-timesteps,0)):
            ax = fig.add_subplot(gs[k,i])
            im = plot_precip_image(ax, R[k,0,j,:,:])
        for (i,j) in enumerate(range(-timesteps,0)):
            ax = fig.add_subplot(gs[k,i+timesteps])
            im = plot_precip_image(ax, R_hat[k,0,j,:,:])
    
    cax = fig.add_subplot(gs[:,-1])
    plt.colorbar(im, cax=cax)

    if out_file is not None:
        out_file = fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def plot_animation(x, y, out_dir, sample=0, fmt="{}_{:02d}.png"):
    def plot_frame(R, label, timestep):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = plot_precip_image(ax, R[sample,0,timestep,:,:])
        fn = fmt.format(label, k)
        fn = os.path.join(out_dir, fn)
        fig.savefig(fn, bbox_inches='tight')
        plt.close(fig)
    
    for k in range(x.shape[2]):
        plot_frame(x, "x", k)
    
    for k in range(y.shape[2]):
        plot_frame(y, "y", k)


model_colors = {
    "mch-dgmr": "#E69F00",
    "mch-pysteps": "#009E73",
    "mch-iters=50-res=256": "#0072B2",
    "dwd-dgmr": "#E69F00",
    "dwd-pysteps": "#009E73",
    "dwd-iters=50-res=256": "#0072B2",
    "pm-mch-dgmr": "#E69F00",
    "pm-mch-pysteps": "#009E73",
    "pm-mch-iters=50-res=256": "#0072B2",
    "pm-dwd-dgmr": "#E69F00",
    "pm-dwd-pysteps": "#009E73",
    "pm-dwd-iters=50-res=256": "#0072B2",
}

scale_linestyles = {
    "1x1": "-",
    "8x8": "--",
    "64x64": ":"
}

def plot_crps(
    log=False,
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    interval_mins=5,
    out_fn=None,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
):
    crps = {}
    crps_name = "logcrps" if log else "crps"
    for model in models:
        crps[model] = {}
        fn = f"../results/crps/{crps_name}-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                var = f"crps_pool{scale}"                
                crps[model][scale] = \
                    np.array(ds[var][:], copy=False).mean(axis=(0,1,3,4))

    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
    
    max_t = 0
    for (model, label) in zip(models, model_labels):
        for scale in scales:
            score = crps[model][scale]
            color = model_colors[model]
            linestyle = scale_linestyles[scale]
            t = np.arange(
                interval_mins, (len(score)+0.1)*interval_mins, interval_mins
            )
            max_t = max(max_t, t[-1])
            ax.plot(t, score, color=color, linestyle=linestyle,
                label=label)

    if add_legend:
        plt.legend()
    if add_xlabel:
        plt.xlabel("Lead time [min]", fontsize=12)
    if add_ylabel:
        plt.ylabel(
            "LogCRPS" if log else "CRPS [mm h$^\\mathrm{-1}$]",
            fontsize=12
        )

    ax.set_xlim((0, max_t))
    ylim = ax.get_ylim()
    ylim = (0, ylim[1])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)


def plot_rank_distribution(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    interval_mins=5,
    out_fn=None,
    num_ensemble_members=32,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
):
    rank_hist = {}
    rank_KL = {}
    for model in models:
        fn = f"../results/ranks/ranks-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                var = f"ranks_pool{scale}"
                ranks = np.array(ds[var][:], copy=False)
                rank_hist[(model,scale)] = rank.rank_distribution(ranks)
                rank_KL[(model,scale)] = rank.rank_DKL(rank_hist[(model,scale)])
                del ranks
    
    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
    
    for scale in scales:        
        linestyle = scale_linestyles[scale]  
        for (model, label) in zip(models, model_labels):        
            h = rank_hist[(model,scale)]
            color = model_colors[model]          
            x = np.linspace(0, 1, num_ensemble_members+1)
            label_with_score = f"{label}: {rank_KL[(model,scale)]:.3f}"
            ax.plot(x, h, color=color, linestyle=linestyle,
                label=label_with_score)
        h_ideal = 1/(num_ensemble_members+1)
        ax.plot([0, 1], [h_ideal, h_ideal], color=(0.4,0.4,0.4), 
            linewidth=1.0)

    if add_legend:
        ax.legend(loc='upper center')
    if add_xlabel:
        ax.set_xlabel("Normalized rank", fontsize=12)
    if add_ylabel:
        ax.set_ylabel("Occurrence", fontsize=12)

    ax.set_xlim((0, 1))
    ylim = ax.get_ylim()
    ylim = (0, ylim[1])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    # int labels for 0 and 1 to save space
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)


def plot_rank_metric(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    interval_mins=5,
    metric_name="KL",
    out_fn=None,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
):
    rank_metric = {}
    for model in models:
        rank_metric[model] = {}
        fn = f"../results/ranks/ranks-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                var = f"ranks_pool{scale}"
                ranks = np.array(ds[var][:], copy=False)
                rank_metric[model][scale] = rank.rank_metric_by_leadtime(ranks)
                del ranks
        
    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
    
    max_t = 0
    for (model, label) in zip(models, model_labels):
        for scale in scales:
            score = rank_metric[model][scale]
            color = model_colors[model]
            linestyle = scale_linestyles[scale]
            label_with_scale = f"{label} {scale}"
            t = np.arange(
                interval_mins, (len(score)+0.1)*interval_mins, interval_mins
            )
            max_t = max(max_t, t[-1])
            ax.plot(t, score, color=color, linestyle=linestyle,
                label=label_with_scale)

    if add_legend:
        plt.legend()
    if add_xlabel:
        plt.xlabel("Lead time [min]", fontsize=12)
    if add_ylabel:
        plt.ylabel(f"Rank {metric_name}", fontsize=12)

    ax.set_xlim((0, max_t))
    ylim = ax.get_ylim()
    ylim = (0, ylim[1])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)


def load_fss(model, scale, use_timesteps):
    fn = f"../results/fractions/fractions-{model}.nc"
    with netCDF4.Dataset(fn, 'r') as ds:
        sn = f"{scale}x{scale}"
        obs_frac = np.array(ds[f"obs_frac_scale{sn}"][:], copy=False)
        fc_frac = np.array(ds[f"fc_frac_scale{sn}"][:], copy=False)
        return fss.fractions_skill_score(
            obs_frac, fc_frac, use_timesteps=use_timesteps
        )


def plot_fss(
    log=False,
    models=("iters=50-res=256", "dgmr", "pysteps"),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    interval_mins=5,
    out_fn=None,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    scales=None,
    use_timesteps=18
):
    if scales is None:
        scales = 2**np.arange(9)
    fss_scale = {}
    N_threads = min(multiprocessing.cpu_count(), len(models)*len(scales))
    with concurrent.futures.ProcessPoolExecutor(N_threads) as executor:
        for model in models:
            fss_scale[model] = {}
            for scale in scales:
                fss_scale[model][scale] = executor.submit(
                    load_fss, model, scale, use_timesteps
                )

        for model in models:
            for scale in scales:
                fss_scale[model][scale] = fss_scale[model][scale].result()

    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
    
    for (model, label) in zip(models, model_labels):
        scales = sorted(fss_scale[model])
        fss_for_model = [fss_scale[model][s] for s in scales]
        
        model_parts = model.split("-")
        if model.startswith("pm-"):
            model_without_threshold = "-".join(model_parts[:1] + model_parts[2:])
        else:
            model_without_threshold = "-".join(model_parts[1:])
        color = model_colors[model_without_threshold]

        ax.plot(scales, fss_for_model, color=color,
            label=label)

    if add_legend:
        plt.legend()
    if add_xlabel:
        plt.xlabel("Scale [km]", fontsize=12)
    if add_ylabel:
        plt.ylabel("FSS")

    ax.set_xlim((scales[0], scales[-1]))
    ylim = ax.get_ylim()
    ylim = (0, ylim[1])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)
