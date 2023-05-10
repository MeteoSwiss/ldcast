import contextlib
import gc
from itertools import chain
import os

from fire import Fire
import torch
from torch import nn
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from ldcast.features.io import save_batch
from ldcast.models.diffusion import plms

import eval_data
import train_genforecast


def io_process(idx, queue, fn_template, out_dir):
    while (data := queue.get()) is not None:
        print(f"IO process {idx} received data")
        (x, y, y_pred, batch_index) = data
        save_batch(x, y, y_pred, batch_index, fn_template, out_dir)
        queue.task_done()


def nested_to(x, **kwargs):
    if isinstance(x, list) or isinstance(x, tuple):
        return [nested_to(xx, **kwargs) for xx in x]
    else:
        return x.to(**kwargs)


def ldm_process(
    idx,
    input_queue,
    output_queue,
    ensemble_size=32,
    weights_fn="../models/genforecast/genforecast-radaronly-256x256-20step.pt",
    num_diffusion_iters=50,
):
    (ldm, trainer) = train_genforecast.setup_model()
    ldm.to(device=f"cuda:{idx}")

    def load_weights(fn):
        state_dict = torch.load(fn, map_location=ldm.device)
        if "state_dict" in state_dict: # loaded a model checkpoint
            state_dict = state_dict["state_dict"]
        ldm.load_state_dict(state_dict, strict=False)
    many_weights = not isinstance(weights_fn, str)
    if not many_weights:
        load_weights(weights_fn)

    sampler = plms.PLMSSampler(ldm)
    print(f"LDM ready at {idx}")

    while (data := input_queue.get()) is not None:
        print(f"Compute process {idx} received data")
        (x, y, batch_index) = data
        gc.collect()
        if many_weights:
            print(f"Compute process {idx} loading {weights_fn[batch_index]}")
            load_weights(weights_fn[batch_index])

        batch_size = y.shape[0]
        x = nested_to(x, device=ldm.device)

        y_pred = []
        gen_shape = (32, 5) + (y.shape[-2]//4, y.shape[-1]//4)
        for member in range(ensemble_size):
            print(f"Compute process {idx} generating member {member+1}/{ensemble_size}")
            with contextlib.redirect_stdout(None):
                (s, intermediates) = sampler.sample(
                    num_diffusion_iters, 
                    batch_size,
                    gen_shape,
                    x,
                    progbar=False
                )
            y_pred.append(ldm.autoencoder.decode(s))
        
        y_pred = torch.stack(y_pred, dim=-1)
        y_pred = nested_to(y_pred, device='cpu')
        x = nested_to(x, device='cpu')

        output_queue.put((x, y, y_pred, batch_index))
        input_queue.task_done()


def create_evaluation_ensemble(
    out_dir="../results/eval_ensembles/genforecast/",
    fn_template="ensemble_batch-{batch_index:04d}.nc",
    resolution=256,
    batch_size=8,
    num_samples=1024,
    ensemble_size=32,
    weights_fn="../models/genforecast/genforecast-radaronly-256x256-20step.pt",
    num_diffusion_iters=50,
    dataset_id="testset"
):
    data_iter = eval_data.get_data_iter(
        dataset_id, resolution=resolution, batch_size=batch_size
    )
    num_batches = num_samples // batch_size
    num_gpus = torch.cuda.device_count()
    num_io_procs = min(num_gpus, mp.cpu_count())

    context = mp.get_context('spawn')
    compute_queue = context.JoinableQueue(num_gpus+1)
    io_queue = context.JoinableQueue(num_io_procs+1)

    compute_procs = mp.spawn(
        ldm_process, 
        args=(
            compute_queue, io_queue,
            ensemble_size, weights_fn, num_diffusion_iters
        ),
        nprocs=num_gpus,
        join=False
    )

    io_procs = mp.spawn(
        io_process,
        args=(io_queue, fn_template, out_dir),
        nprocs=num_io_procs,
        join=False
    )

    os.makedirs(out_dir, exist_ok=True)

    for batch_idx in range(num_batches):
        print(f"Sending batch {batch_idx+1}/{num_batches}")
        (x,y) = next(data_iter)
        compute_queue.put((x, y, batch_idx))

    compute_queue.join()
    io_queue.join()
    for i in range(num_gpus):
        compute_queue.put(None)    
    for i in range(num_io_procs):
        io_queue.put(None)
    while not compute_procs.join():
        pass
    while not io_procs.join():
        pass


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    create_evaluation_ensemble(**config)


if __name__ == "__main__":
    Fire(main)
