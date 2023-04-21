LDCast is a precipitation nowcasting model based on a latent diffusion model (LDM, used by e.g. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)).

This repository contains the code for using LDCast to make predictions and the code used to generate the analysis in the LDCast paper (a preprint will be available soon).

A GPU is recommended for both using and training LDCast, although you may be able to generate some samples with a CPU and enough patience.

# Installation

Clone the repository, then, in the main directory, run
```bash
$ pip install -e .
```
This should automatically install the required packages. If you don't want that, use:
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

fc = forecast.Forecast(ldm_weights_fn, autoenc_weights_fn)
R_pred = fc(R_past)
```
Here, `ldm_weights_fn` is the path to the LDM weights and `autoenc_weights_fn` is the path to the autoencoder weights. `R_past` is a NumPy array of precipitation rates with shape `(timesteps, height, width)` where `timesteps` must be 4 and `height` and `width` must be divisible by 32.

## Demo

For a practical example, you can run the demo in the `scripts` directory. First download the `ldcast-demo-20210622.zip` file from the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914), then unzip it in the `data` directory. Then run
```bash
$ python forecast_demo.py
```
A sample output can be found in the file `ldcast-demo-video-20210622.zip` in the data repository. See the function `forecast_demo` in `forecast_demo.py` see how the `Forecast` class works.

# Training 

## Training data

The preprocessed training data, needed to rerun the LDCast training, can be found at the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914). Unzip the `ldcast-datasets.zip` file to the `data` directory.

## Training the autoencoder

In the `scripts` directory, run
```bash
$ python train_autoenc.py
```
to run the training of the autoencoder with the default parameters.

## Training the diffusion model

In the `scripts` directory, run
```bash
$ python train_genforecast.py
```
to run the training of the diffusion model with the default parameters, or
```bash
$ python train_genforecast.py --config=<path_to_config_file>
```
to run the training with different parameters. Some config files can be found in the `config` directory.

# Evaluation

You can find scripts for evaluating models in the `scripts` directory:
* `eval_genforecast.py` to evaluate LDCast
* `eval_dgmr.py` to evaluate DGMR (requires tensorflow installation and the DGMR model from https://github.com/deepmind/deepmind-research/tree/master/nowcasting placed in the `models/dgmr` directory)
* `eval_pysteps.py` to evaluate PySTEPS (requires pysteps installation)
* `metrics.py` to produce metrics from the evaluation results produced with the functions in scripts above
* `plot_genforecast.py` to make plots from the results generated

