LDCast is a precipitation nowcasting model based on a latent diffusion model (LDM, used by e.g. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)).

This repository contains the code for using LDCast to make predictions and the code used to generate the analysis in the LDCast paper (a preprint is available at https://arxiv.org/abs/2304.12891).

A GPU is recommended for both using and training LDCast, although you may be able to generate some samples with a CPU and enough patience.

# Installation

It is recommended you install the code in its own virtual environment (created with e.g. pyenv or conda).

Clone the repository, then, in the main directory, run
```bash
$ pip install -e .
```
This should automatically install the required packages (which might take some minutes). In the paper, we used PyTorch 11.2 but are not aware of any problems with newer versions.

If you don't want the requirements to be installed (e.g. if you installed them manually with conda), use:
```bash
$ pip install --no-dependencies -e .
```

# Using LDCast

## Pretrained models

The pretrained models are available at the Zenodo repository https://doi.org/10.5281/zenodo.7780914. Unzip the file `ldcast-models.zip`. The default is to unzip it to the `models` directory, but you can also use another location.

## Producing predictions

The easiest way to produce predictions is to use the `ldcast.forecast.Forecast` class, which will set up all models and data transformations and is callable with a past precipitation array.
```python
from ldcast import forecast

fc = forecast.Forecast(
    ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn
)
R_pred = fc(R_past)
```
Here, `ldm_weights_fn` is the path to the LDM weights and `autoenc_weights_fn` is the path to the autoencoder weights. `R_past` is a NumPy array of precipitation rates with shape `(timesteps, height, width)` where `timesteps` must be 4 and `height` and `width` must be divisible by 32.

### Ensemble predictions

If want to process multiple cases at once and/or generate several ensemble members, there is the `ldcast.forecast.ForecastDistributed` class. The usage is similar to the `Forecast` class, for example:
```python
from ldcast import forecast

fc = forecast.ForecastDistributed(
    ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn,
    ensemble_members=32
)
R_pred = fc(R_past)
```
Here, `R_past` should be of shape `(cases, timesteps, height, width)` where `cases` is the number of cases you want to process. For each case, `ensemble_members` predictions are produced (this is the last axis of `R_pred`). `ForecastDistributed` automatically distributes the workload to multiple GPUs if you have them.

## Demo

For a practical example, you can run the demo in the `scripts` directory. First download the `ldcast-demo-20210622.zip` file from the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914), then unzip it in the `data` directory. Then run
```bash
$ python forecast_demo.py
```
A sample output can be found in the file `ldcast-demo-video-20210622.zip` in the data repository. See the function `forecast_demo` in `forecast_demo.py` see how the `Forecast` class works. To run an ensemble mean of 8 members using the `ForecastDistributed` class, you can use:
```bash
$ python forecast_demo.py --ensemble-members=8
```

The demo for a single ensemble member runs in a couple of minutes on our system using one V100 GPU; with a CPU around 10 minutes or more would be expected. A progress bar will show the status of the generation.

# Training 

## Training data

The preprocessed training data, needed to rerun the LDCast training, can be found at the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914). Unzip the `ldcast-datasets.zip` file to the `data` directory.

## Training the autoencoder

In the `scripts` directory, run
```bash
$ python train_autoenc.py --model_dir="../models/autoenc_train"
```
to run the training of the autoencoder with the default parameters. The training checkpoints will be saved in the `../models/autoenc_train` directory (feel free to change this).

It has been reported that this training may encounter a condition where the loss goes to `nan`. If this happens, try restarting from the latest checkpoint:
```bash
$ python train_autoenc.py --model_dir="../models/autoenc_train" --ckpt_path="../models/autoenc_train/<checkpoint_file>"
```
where `<checkpoint_file>` should be the latest checkpoint in the `../models/autoenc_train/` directory.

## Training the diffusion model

In the `scripts` directory, run
```bash
$ python train_genforecast.py --model_dir="../models/genforecast_train"
```
to run the training of the diffusion model with the default parameters, or
```bash
$ python train_genforecast.py --model_dir="../models/genforecast_train" --config=<path_to_config_file>
```
to run the training with different parameters. Some config files can be found in the `config` directory. The training checkpoints will be saved in the `../models/genforecast_train` directory (again, this can be changed freely).

# Evaluation

You can find scripts for evaluating models in the `scripts` directory:
* `eval_genforecast.py` to evaluate LDCast
* `eval_dgmr.py` to evaluate DGMR (requires tensorflow installation and the DGMR model from https://github.com/deepmind/deepmind-research/tree/master/nowcasting placed in the `models/dgmr` directory)
* `eval_pysteps.py` to evaluate PySTEPS (requires pysteps installation)
* `metrics.py` to produce metrics from the evaluation results produced with the functions in scripts above
* `plot_genforecast.py` to make plots from the results generated
