import gc
import os

from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.io import save_batch
from ldcast.models.benchmarks.pysteps import PySTEPSModel
from ldcast.models.benchmarks.transform import transform_to_rainrate, transform_from_rainrate

import eval_data


def create_evaluation_ensemble(
    out_dir="../results/eval_ensembles/pysteps/",
    fn_template="ensembles-pysteps-{batch_index:04d}.nc",
    resolution=256,
    batch_size=8,
    num_samples=1024,
    ensemble_size=32,
    dataset_id="testset"
):
    data_iter = eval_data.get_data_iter(
        dataset_id, resolution=resolution, batch_size=batch_size
    )
    num_batches = num_samples // batch_size

    pysteps = PySTEPSModel(
        transform_to_rainrate=transform_to_rainrate,
        transform_from_rainrate=transform_from_rainrate,
        ensemble_size=ensemble_size
    )

    os.makedirs(out_dir, exist_ok=True)

    for batch_idx in range(num_batches):
        print(f"Sending batch {batch_idx+1}/{num_batches}")
        (x,y) = next(data_iter)
        y_pred = pysteps(x)
        save_batch(
            x, y, y_pred, batch_idx, fn_template, out_dir
        )
        gc.collect()


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    create_evaluation_ensemble(**config)


if __name__ == "__main__":
    Fire(main)
