from itertools import chain, product
import os

from matplotlib import gridspec, pyplot as plt
import numpy as np

from ldcast.features import io
from ldcast.visualization import plots


def plot_crps(
    scales=("1x1","8x8","64x64"),
    model_sets=(
        ("pm-mch-iters=50-res=256", "pm-mch-dgmr", "pm-mch-pysteps", "mch-persistence"),
        ("pm-dwd-iters=50-res=256", "pm-dwd-dgmr", "pm-dwd-pysteps", "dwd-persistence"),
        ("pm-mch-iters=50-res=256", "pm-mch-dgmr", "pm-mch-pysteps", "mch-persistence"),
        ("pm-dwd-iters=50-res=256", "pm-dwd-dgmr", "pm-dwd-pysteps", "dwd-persistence"),
    ),
    dataset_labels=("Switzerland", "Germany", "Switzerland", "Germany"),
    xticks=(0,30,60,90),
    log=(False, False, True, True),
    out_fn="../figures/crps-leadtime.pdf",
    crop_box=None
):
    N_modelsets = len(model_sets)
    N_scales = len(scales)

    fig = plt.figure(figsize=(3.2 * N_scales, 2.0 * N_modelsets), dpi=150)
    gs = gridspec.GridSpec(N_modelsets, N_scales, wspace=0.05, hspace=0.07)

    for (i,models) in enumerate(model_sets):
        for (j, scale) in enumerate(scales):
            ax = fig.add_subplot(gs[i,j])
            plots.plot_crps(
                ax=ax,
                log=log[i], models=models, scales=(scale,), crop_box=crop_box,
                add_xlabel=(i==N_modelsets-1),
                add_ylabel=(j==0),
                add_legend=(i==j==0)
            )
            if (j==0):
                ylim = ax.get_ylim()
                ax.set_ylabel(dataset_labels[i]+"\n"+ax.get_ylabel())
            else:
                ax.set_ylim(ylim)
                ax.tick_params(labelleft=False)
            if (i==0):
                ax.set_title(scale)
            if (i < N_modelsets-1):
                ax.tick_params(labelbottom=False)
            ax.set_xticks(xticks)
            
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_rank(
    scales=("1x1","8x8","64x64"),
    model_sets=(
        ("pm-mch-iters=50-res=256", "pm-mch-dgmr", "pm-mch-pysteps"),
        ("pm-dwd-iters=50-res=256", "pm-dwd-dgmr", "pm-dwd-pysteps"),
    ),
    dataset_labels=("Switzerland", "Germany"),
    out_fn="../figures/rank-distribution.pdf",
    crop_box=None
):
    N_modelsets = len(model_sets)
    N_scales = len(scales)

    fig = plt.figure(figsize=(3.2 * N_scales, 2.0 * N_modelsets), dpi=150)
    gs = gridspec.GridSpec(N_modelsets, N_scales, wspace=0.05, hspace=0.07)

    for (i,models) in enumerate(model_sets):
        for (j, scale) in enumerate(scales):
            ax = fig.add_subplot(gs[i,j])
            plots.plot_rank_distribution(
                ax=ax,
                models=models, scales=(scale,), crop_box=crop_box,
                add_xlabel=(i==N_modelsets-1),
                add_ylabel=(j==0),
                add_legend=True
            )
            if (j==0):
                ylim = ax.get_ylim()
                ax.set_ylabel(dataset_labels[i]+"\n"+ax.get_ylabel())
            else:
                ax.set_ylim(ylim)
                ax.tick_params(labelleft=False)
            if (i==0):
                ax.set_title(scale)
            if (i < N_modelsets-1):
                ax.tick_params(labelbottom=False)
            
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_rank_metric(
    scales=("1x1","8x8","64x64"),
    model_sets=(
        ("pm-mch-iters=50-res=256", "pm-mch-dgmr", "pm-mch-pysteps"),
        ("pm-dwd-iters=50-res=256", "pm-dwd-dgmr", "pm-dwd-pysteps"),
    ),
    dataset_labels=("Switzerland", "Germany"),
    xaxis="leadtime",
    out_fn="../figures/rank-metric.pdf"
):
    N_modelsets = len(model_sets)
    N_scales = len(scales)

    if xaxis == "leadtime":
        xticks = (0, 30, 60, 90)
        plot_func = plots.plot_rank_metric
    elif xaxis == "bins":
        xticks = (0.1, 0.3, 1, 3, 10, 30)
        plot_func = plots.plot_rank_conditional

    fig = plt.figure(figsize=(3.2 * N_scales, 2.0 * N_modelsets), dpi=150)
    gs = gridspec.GridSpec(N_modelsets, N_scales, wspace=0.05, hspace=0.07)

    for (i,models) in enumerate(model_sets):
        for (j, scale) in enumerate(scales):
            ax = fig.add_subplot(gs[i,j])
            plots.plot_rank_metric(
                ax=ax,
                models=models, scales=(scale,),
                add_xlabel=(i==N_modelsets-1),
                add_ylabel=(j==0),
                add_legend=True
            )
            if (j==0):
                ax.set_ylabel(dataset_labels[i]+"\n"+ax.get_ylabel())
            else:
                ax.tick_params(labelleft=False)
            if (i==0):
                ax.set_title(scale)
            if (i < N_modelsets-1):
                ax.tick_params(labelbottom=False)
            ax.set_xticks(xticks)
            
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_fss(
    thresholds=("0.1", "1.0", "10.0"),
    model_sets=(
        ("pm-{T}-mch-iters=50-res=256", "pm-{T}-mch-dgmr", "pm-{T}-mch-pysteps"),
        ("pm-{T}-dwd-iters=50-res=256", "pm-{T}-dwd-dgmr", "pm-{T}-dwd-pysteps"),
    ),
    dataset_labels=("Switzerland", "Germany"),
    xticks=(1, 50, 100, 150, 200),
    out_fn="../figures/fss-threshold.pdf",
    crop_box=None
):
    N_modelsets = len(model_sets)
    N_thresholds = len(thresholds)

    fig = plt.figure(figsize=(3.2 * N_thresholds, 2.0 * N_modelsets), dpi=150)
    gs = gridspec.GridSpec(N_modelsets, N_thresholds, wspace=0.05, hspace=0.07)

    for (i, models) in enumerate(model_sets):
        for (j, T) in enumerate(thresholds):
            ax = fig.add_subplot(gs[i,j])
            models_T = [m.format(T=T) for m in models]
            plots.plot_fss(
                ax=ax,
                models=models_T,
                add_xlabel=(i==N_modelsets-1),
                add_ylabel=(j==0),
                add_legend=(i==j==0),
                crop_box=crop_box
            )
            if (j==0):
                ylim = ax.get_ylim()
                ax.set_ylabel(dataset_labels[i]+"\n"+ax.get_ylabel())
            else:
                ax.set_ylim(ylim)
                ax.tick_params(labelleft=False)
            if (i==0):
                ax.set_title(f"T={T} mm h$^\\mathrm{{-1}}$")
            if (i < N_modelsets-1):
                ax.tick_params(labelbottom=False)
            ax.set_xticks(xticks)
            
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_csi(
    thresholds=("0.1", "1.0", "10.0"),
    scale="1x1",
    model_sets=(
        ("pm-{T}-mch-iters=50-res=256", "pm-{T}-mch-dgmr", "pm-{T}-mch-pysteps"),
        ("pm-{T}-dwd-iters=50-res=256", "pm-{T}-dwd-dgmr", "pm-{T}-dwd-pysteps"),
    ),
    dataset_labels=("Switzerland", "Germany"),
    out_fn=None,
    plot_by="leadtime",
    crop_box=None
):
    N_modelsets = len(model_sets)
    N_thresholds = len(thresholds)

    fig = plt.figure(figsize=(3.2 * N_thresholds, 2.0 * N_modelsets), dpi=150)
    gs = gridspec.GridSpec(N_modelsets, N_thresholds, wspace=0.05, hspace=0.07)

    for (i, models) in enumerate(model_sets):
        for (j, T) in enumerate(thresholds):
            ax = fig.add_subplot(gs[i,j])
            models_T = [m.format(T=T) for m in models]
            plot_func = {
                "leadtime": plots.plot_csi_leadtime,
                "threshold": plots.plot_csi_threshold
            }[plot_by]
            plot_func(
                ax=ax,
                scales=(scale,),
                models=models_T,
                add_xlabel=(i==N_modelsets-1),
                add_ylabel=(j==0),
                add_legend=(i==j==0),
                crop_box=crop_box
            )
            if (j==0):
                ylim = ax.get_ylim()
                ax.set_ylabel(dataset_labels[i]+"\n"+ax.get_ylabel())
            else:
                ax.set_ylim(ylim)
                ax.tick_params(labelleft=False)
            if (i==0):
                ax.set_title(f"T={T} mm h$^\\mathrm{{-1}}$")
            if (i < N_modelsets-1):
                ax.tick_params(labelbottom=False)

    if out_fn is None:
        out_fn = f"../figures/csi-{plot_by}.pdf"      
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_sample(
    models=("iters=50-res=256", "dgmr", "pysteps"),
    sample=122,
    path="../results/eval_ensembles",
    samples_per_batch=8,
    past_timesteps=(-3,),
    future_timesteps=(0,3,7,11,15),
    ensemble_members=(0,),
    out_fn="../figures/samples.pdf"
):
    data = {}
    batch = sample // samples_per_batch
    sample = sample % samples_per_batch
    for model in models:
        fn = f"ensembles-{model}-{batch:04d}.nc"
        fn = os.path.join(path, model, fn)
        (x, y, y_pred) = io.load_batch(fn)
        x = x[sample,...]
        y = y[sample,...]
        y_pred = y_pred[sample,...]
        data[model] = (x, y, y_pred)

    first_model = models[0]
    (x, y, y_pred) = data[first_model]
    N_timesteps = len(past_timesteps) + len(future_timesteps)
    N_models = len(models)
    N_total_members = N_models * len(ensemble_members)
    fig = plt.figure(figsize=(1.5*N_timesteps, 1.5*(N_total_members+1)), dpi=300)
    gs = gridspec.GridSpec(N_total_members+1, N_timesteps, wspace=0.05, hspace=0.05)

    for (j,t) in enumerate(past_timesteps):
        ax = fig.add_subplot(gs[0,j])
        plots.plot_precip_image(ax, x[0,t,:,:], transform_R=False)
    for (i,(model,member)) in enumerate(chain((("obs", 0),),product(models,ensemble_members))):
        for (j,t) in enumerate(future_timesteps, start=len(past_timesteps)):
            ax = fig.add_subplot(gs[i,j])
            if model == "obs":
                plots.plot_precip_image(ax, y[0,t,:,:], transform_R=False)
            else:
                plots.plot_precip_image(ax, y_pred[0,t,:,:,member],  transform_R=False)
            
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_sample_cases(
    models=(
        "mch-iters=50-res=256",
        "mch-iters=50-res=256",
        "dwd-iters=50-res=256",
        "dwd-iters=50-res=256"
    ),
    samples=(28,112,20,75),
    path="../results/eval_ensembles",
    samples_per_batch=8,
    past_timesteps=(-1,),
    future_timesteps=(3,7,11,15,19),
    ensemble_member=0,
    out_fn="../figures/sample-cases.pdf",
    timestep_mins=5
):
    data = []
    for (model,sample) in zip(models,samples):
        batch = sample // samples_per_batch
        batch_sample = sample % samples_per_batch
        fn = f"ensembles-{model}-{batch:04d}.nc"
        fn = os.path.join(path, model, fn)
        (x, y, y_pred) = io.load_batch(fn)
        x = x[batch_sample,...]
        y = y[batch_sample,...]
        y_pred = y_pred[batch_sample,...]
        data.append((x, y, y_pred))

    N_timesteps = len(past_timesteps) + len(future_timesteps)
    N_models = len(models)
    N_rows = 2 * N_models
    fig = plt.figure(figsize=(1.5*N_timesteps, 1.5*N_rows), dpi=300)
    gs_outer = gridspec.GridSpec(N_models, 2, 
        wspace=0.02, hspace=0.05, width_ratios=(0.98,0.02))

    for (k,(x,y,y_pred)) in enumerate(data):
        gs = gridspec.GridSpecFromSubplotSpec(
            2, N_timesteps, subplot_spec=gs_outer[k,0],
            wspace=0.02, hspace=0.02
        )
        for (j,t) in enumerate(past_timesteps):
            ax = fig.add_subplot(gs[0,j])
            plots.plot_precip_image(ax, x[0,t,:,:], transform_R=False)
            if t == past_timesteps[0]:
                ax.set_ylabel("Observation")
            if k==0:
                t_min = (t+1) * timestep_mins
                timestamp = f"{t_min:+d} min"
                ax.set_title(timestamp, fontsize=10)
        for (j,t) in enumerate(future_timesteps, start=len(past_timesteps)):
            ax = fig.add_subplot(gs[0,j])
            plots.plot_precip_image(ax, y[0,t,:,:], transform_R=False)
            if k==0:
                t_min = (t+1) * timestep_mins
                timestamp = f"{t_min:+d} min"
                ax.set_title(timestamp, fontsize=10)
            ax = fig.add_subplot(gs[1,j])
            im = plots.plot_precip_image(
                ax, y_pred[0,t,:,:,ensemble_member], transform_R=False
            )
            if t == future_timesteps[0]:
                ax.set_ylabel("LDCast")

    cax = fig.add_subplot(gs_outer[0,1])
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_title("$\\mathrm{mm\\,h^{-1}}$", fontsize=10)
                
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_sample_cases_all(
    first_sample=512,
    num_samples=256,
    region="mch", # mch or dwd
):
    out_dir = f"../figures/sample-cases/{region}"
    os.makedirs(out_dir, exist_ok=True)
    model = f"{region}-iters=50-res=256"
    for k in range(num_samples):
        fn = f"sample-case-{k:04d}.pdf"
        print(fn)
        fn = os.path.join(out_dir, fn)
        plot_sample_cases(samples=(first_sample+k,), models=(model,), out_fn=fn)


def plot_ensemble(
    models=(
        ("mch-iters=50-res=256", "mch-dgmr"),
        ("mch-iters=50-res=256", "mch-dgmr"),
        ("dwd-iters=50-res=256", "dwd-dgmr"),
        ("dwd-iters=50-res=256", "dwd-dgmr")
    ),
    samples=(89,17,58,119),
    path="../results/eval_ensembles",
    samples_per_batch=8,
    future_timestep=17,
    ensemble_members=(0,1,2,3,4),
    out_fn="../figures/ensemble-diversity.pdf",
    timestep_mins=5
):
    data = {}
    for (model_list,sample) in zip(models,samples):
        batch = sample // samples_per_batch
        batch_sample = sample % samples_per_batch
        for model in model_list:
            fn = f"ensembles-{model}-{batch:04d}.nc"
            fn = os.path.join(path, model, fn)
            (x, y, y_pred) = io.load_batch(fn)
            x = x[batch_sample,...]
            y = y[batch_sample,...]
            y_pred = y_pred[batch_sample,...]
            data[(model,sample)] = (x, y, y_pred)

    N_cases = len(models)
    N_rows = sum(len(m) for m in models)
    N_members = len(ensemble_members)
    fig = plt.figure(figsize=(1.5*(N_members+1), 1.5*N_rows), dpi=300)
    gs_outer = gridspec.GridSpec(N_cases, 2, 
        wspace=0.02, hspace=0.05, width_ratios=(0.98,0.02))

    t = future_timestep
    for (k, (model_list,sample)) in enumerate(zip(models,samples)):        
        gs = gridspec.GridSpecFromSubplotSpec(
            2*len(model_list), N_members+1, subplot_spec=gs_outer[k,0],
            wspace=0.02, hspace=0.02
        )
        for (i,model) in enumerate(model_list):
            (x, y, y_pred) = data[(model,sample)]

            for (j,member) in enumerate(ensemble_members):
                fc = y_pred[0,t,:,:,member]
                ax = fig.add_subplot(gs[i*2:i*2+2,j])
                im = plots.plot_precip_image(ax, y_pred[0,t,:,:,member], transform_R=False)
                if j==0:
                    ax.set_ylabel("DGMR" if model.endswith("dgmr") else "LDCast")

            if i == 0:
                i_obs = len(model_list)-1
                ax = fig.add_subplot(gs[i_obs:i_obs+2,-1])
                plots.plot_precip_image(ax, y[0,t,:,:], transform_R=False)
                ax.set_title("Observation", fontsize=10)

    cax = fig.add_subplot(gs_outer[0,1])
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_title("$\\mathrm{mm\\,h^{-1}}$", fontsize=10)
                
    fig.savefig(out_fn, bbox_inches='tight')
    plt.close(fig)


def plot_ensemble_all(
    first_sample=512,
    num_samples=256,
    region="mch", # mch or dwd
):
    out_dir = f"../figures/ensemble-diversity/{region}"
    os.makedirs(out_dir, exist_ok=True)
    models = (f"{region}-iters=50-res=256", f"{region}-dgmr")
    for k in range(num_samples):
        fn = f"ensemble-diversity-{k:04d}.pdf"
        print(fn)
        fn = os.path.join(out_dir, fn)
        plot_ensemble(samples=(first_sample+k,), models=(models,), out_fn=fn)


def plot_all():
    plot_crps()
    plot_rank()
    plot_fss()
