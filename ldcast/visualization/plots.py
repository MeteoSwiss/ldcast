import concurrent
import multiprocessing
import netCDF4
import os

from matplotlib import gridspec, colors, pyplot as plt
import numpy as np
import torch

from ..analysis import confmatrix, fss, rank
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
    "mch-persistence": "#888888",
    "dwd-dgmr": "#E69F00",
    "dwd-pysteps": "#009E73",
    "dwd-iters=50-res=256": "#0072B2",
    "dwd-persistence": "#888888",
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
    model_labels=("LDCast", "DGMR", "PySTEPS", "Persist."),
    interval_mins=5,
    out_fn=None,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    crop_box=None
):
    crps = {}
    crps_name = "logcrps" if log else "crps"
    for model in models:
        crps[model] = {}
        fn = f"../results/crps/{crps_name}-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                var = f"crps_pool{scale}"
                crps_model_scale = np.array(ds[var][:], copy=False)
                if crop_box is not None:
                    scale_int = int(scale.split("x")[0])
                    crps_model_scale = crps_model_scale[
                        ...,
                        crop_box[0][0]//scale_int:crop_box[0][1]//scale_int,
                        crop_box[1][0]//scale_int:crop_box[1][1]//scale_int
                    ]
                crps[model][scale] = crps_model_scale.mean(axis=(0,1,3,4))
                del crps_model_scale

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
    out_fn=None,
    num_ensemble_members=32,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    crop_box=None
):
    rank_hist = {}
    rank_KL = {}
    for model in models:
        fn = f"../results/ranks/ranks-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                var = f"ranks_pool{scale}"
                ranks_model_scale = np.array(ds[var][:], copy=False)
                if crop_box is not None:
                    scale_int = int(scale.split("x")[0])
                    ranks_model_scale = ranks_model_scale[
                        ...,
                        crop_box[0][0]//scale_int:crop_box[0][1]//scale_int,
                        crop_box[1][0]//scale_int:crop_box[1][1]//scale_int
                    ]
                rank_hist[(model,scale)] = rank.rank_distribution(ranks_model_scale)
                rank_KL[(model,scale)] = rank.rank_DKL(rank_hist[(model,scale)])
                del ranks_model_scale
    
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


def load_fss(model, scale, use_timesteps, crop_box):
    fn = f"../results/fractions/fractions-{model}.nc"
    with netCDF4.Dataset(fn, 'r') as ds:
        sn = f"{scale}x{scale}"
        obs_frac = np.array(ds[f"obs_frac_scale{sn}"][:], copy=False)
        fc_frac = np.array(ds[f"fc_frac_scale{sn}"][:], copy=False)
        if crop_box is not None:
            obs_frac = obs_frac[
                ...,
                crop_box[0][0]//scale:crop_box[0][1]//scale,
                crop_box[1][0]//scale:crop_box[1][1]//scale
            ]
            fc_frac = fc_frac[
                ...,
                crop_box[0][0]//scale:crop_box[0][1]//scale,
                crop_box[1][0]//scale:crop_box[1][1]//scale
            ]
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
    use_timesteps=18,
    crop_box=None
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
                    load_fss, model, scale, use_timesteps, crop_box
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


def plot_csi_threshold(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    prob_thresholds=tuple(np.linspace(0,1,33)),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    out_fn=None,
    num_ensemble_members=32,
    max_timestep=18,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    crop_box=None
):
    csi = {}
    for model in models:
        fn = f"../results/fractions/fractions-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                fc_var = f"fc_frac_scale{scale}"
                fc_frac = np.array(ds[fc_var], copy=False)
                fc_frac = fc_frac[...,:max_timestep,:,:]
                obs_var = f"obs_frac_scale{scale}"
                obs_frac = np.array(ds[obs_var], copy=False)
                obs_frac = obs_frac[...,:max_timestep,:,:]
                conf_matrix = confmatrix.confusion_matrix_thresholds(
                    fc_frac, obs_frac, prob_thresholds
                )
                del fc_frac, obs_frac
                csi_scale = confmatrix.intersection_over_union(conf_matrix)
                csi[(model,scale)] = csi_scale
    
    if ax is None:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot()
    
    for scale in scales:        
        linestyle = scale_linestyles[scale]  
        for (model, label) in zip(models, model_labels):        
            c = csi[(model,scale)]
            model_parts = model.split("-")
            if model.startswith("pm-"):
                model_without_threshold = "-".join(model_parts[:1] + model_parts[2:])
            else:
                model_without_threshold = "-".join(model_parts[1:])
            color = model_colors[model_without_threshold]          
            ax.plot(prob_thresholds, c, color=color, linestyle=linestyle, label=label)

    if add_legend:
        ax.legend(loc='upper center')
    if add_xlabel:
        ax.set_xlabel("Prob. threshold", fontsize=12)
    if add_ylabel:
        ax.set_ylabel("CSI", fontsize=12)

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


def plot_csi_leadtime(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    prob_thresholds=tuple(np.linspace(0,1,33)),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    out_fn=None,
    interval_mins=5,
    num_ensemble_members=32,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    crop_box=None
):
    csi = {}
    for model in models:
        fn = f"../results/fractions/fractions-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                fc_var = f"fc_frac_scale{scale}"
                fc_frac = np.array(ds[fc_var], copy=False)
                obs_var = f"obs_frac_scale{scale}"
                obs_frac = np.array(ds[obs_var], copy=False)
                conf_matrix = confmatrix.confusion_matrix_thresholds_leadtime(
                    fc_frac, obs_frac, prob_thresholds
                )
                
                csi_scale = confmatrix.intersection_over_union(conf_matrix)
                csi[(model,scale)] = np.nanmax(csi_scale, axis=1)
    
    max_t = 0
    for (model, label) in zip(models, model_labels):
        for scale in scales:
            score = csi[(model,scale)]

            model_parts = model.split("-")
            if model.startswith("pm-"):
                model_without_threshold = "-".join(model_parts[:1] + model_parts[2:])
            else:
                model_without_threshold = "-".join(model_parts[1:])
            color = model_colors[model_without_threshold]            
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
        plt.ylabel("CSI", fontsize=12)

    ax.set_xlim((0, max_t))
    ylim = ax.get_ylim()
    ylim = (0, ylim[1])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)


def plot_cost_loss_value(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    scales=("1x1", "8x8", "64x64"),
    prob_thresholds=tuple(np.linspace(0,1,33)),
    model_labels=("LDCast", "DGMR", "PySTEPS"),
    out_fn=None,
    interval_mins=5,
    num_ensemble_members=32,
    ax=None,
    add_xlabel=True,
    add_ylabel=True,
    add_legend=True,
    crop_box=None
):
    value = {}
    loss = 1.0
    cost = np.linspace(0.01, 1, 100)
    for model in models:
        fn = f"../results/fractions/fractions-{model}.nc"
        with netCDF4.Dataset(fn, 'r') as ds:
            for scale in scales:
                fc_var = f"fc_frac_scale{scale}"
                fc_frac = np.array(ds[fc_var], copy=False)
                obs_var = f"obs_frac_scale{scale}"
                obs_frac = np.array(ds[obs_var], copy=False)
                conf_matrix = confmatrix.confusion_matrix_thresholds(
                    fc_frac, obs_frac, prob_thresholds
                )                

                p_clim = obs_frac.mean()
                value_scale = []
                for c in cost:
                    v = confmatrix.cost_loss_value(
                        conf_matrix, c, loss, p_clim
                    )
                    value_scale.append(v[len(v)//2])
                value[(model,scale)] = np.array(value_scale)
  
    max_score = 0
    for (model, label) in zip(models, model_labels):
        for scale in scales:
            score = value[(model,scale)]
            max_score = max(max_score, score[np.isfinite(score)].max())

            model_parts = model.split("-")
            if model.startswith("pm-"):
                model_without_threshold = "-".join(model_parts[:1] + model_parts[2:])
            else:
                model_without_threshold = "-".join(model_parts[1:])
            color = model_colors[model_without_threshold]            
            linestyle = scale_linestyles[scale]

            ax.plot(cost, score, color=color, linestyle=linestyle,
                label=label)

    if add_legend:
        plt.legend()
    if add_xlabel:
        plt.xlabel("Cost/loss ratio", fontsize=12)
    if add_ylabel:
        plt.ylabel("Value", fontsize=12)

    ax.set_xlim((0, 1))
    ylim = (0, max_score*1.05)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    # int labels for 0 and 1 to save space
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)