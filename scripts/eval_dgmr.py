from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.io import save_batch
from ldcast.models.benchmarks.dgmr import DGMRModel, create_ensemble
from ldcast.models.benchmarks.transform import transform_to_rainrate, transform_from_rainrate

import eval_data


def create_evaluation_ensemble(
    out_dir="../results/eval_ensembles/dgmr/",
    fn_template="ensembles-dgmr-{batch_index:04d}.nc",
    resolution=256,
    batch_size=8,
    num_samples=1024,
    ensemble_size=32,
    model_path="../models/dgmr/256x256/",
    dataset_id="testset",
    calibrated=False
):
    data_iter = eval_data.get_data_iter(
        dataset_id, resolution=resolution, batch_size=batch_size
    )
    num_batches = num_samples // batch_size

    dgmr = DGMRModel(
        model_path,
        transform_to_rainrate=transform_to_rainrate,
        transform_from_rainrate=transform_from_rainrate,
        calibrated=calibrated
    )

    for batch_idx in range(num_batches):
        print(f"Sending batch {batch_idx+1}/{num_batches}")
        (x,y) = next(data_iter)
        y_pred = create_ensemble(dgmr, x)
        y = y[:,:,:y_pred.shape[2],...]
        save_batch(
            x, y, y_pred, batch_idx, fn_template, out_dir
        )


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)
    create_evaluation_ensemble(**config)


if __name__ == "__main__":
    Fire(main)
