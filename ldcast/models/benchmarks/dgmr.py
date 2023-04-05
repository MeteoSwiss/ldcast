import gc

import numpy as np
import tensorflow as tf


class DGMRModel:
    def __init__(
        self, 
        model_handle,
        multi_gpu=True,
        transform_to_rainrate=None,
        transform_from_rainrate=None,
        data_format='channels_first',
        calibrated=False,
    ):
        self.transform_to_rainrate = transform_to_rainrate
        self.transform_from_rainrate = transform_from_rainrate
        self.data_format = data_format
        self.calibrated = calibrated

        if multi_gpu and len(tf.config.list_physical_devices('GPU')) > 1:
            # initialize multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
        else: # use default strategy
            strategy = tf.distribute.get_strategy()
    
        with strategy.scope():
            module = tf.saved_model.load(model_handle)
        
        self.model = module.signatures['default']
        input_signature = self.model.structured_input_signature[1]
        self.noise_dim = input_signature['z'].shape[1]
        self.past_timesteps = input_signature['labels$cond_frames'].shape[1]

    def __call__(self, x):        
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        x = np.array(x, copy=False)
        if self.data_format == "channels_first":
            x = x.transpose(0,2,3,4,1)
        if self.transform_to_rainrate is not None:
            x = self.transform_to_rainrate(x)        
        x = tf.convert_to_tensor(x)

        num_samples = x.shape[0]
        z = tf.random.normal(shape=(num_samples, self.noise_dim))
        if self.calibrated:
            z = z * 2.0

        onehot = tf.ones(shape=(num_samples, 1))
        inputs = {
            "z": z,
            "labels$onehot" : onehot,
            "labels$cond_frames" : x
        }
        y = self.model(**inputs)['default']
        y = y[:,self.past_timesteps:,...]

        y = np.array(y)
        if self.transform_from_rainrate is not None:
            y = self.transform_from_rainrate(y)
        if self.data_format == "channels_first":
            y = y.transpose(0,4,1,2,3)

        return y


def create_ensemble(
    dgmr, x,
    ensemble_size=32,
    model_path="../models/dgmr/256x256",

):
    y_pred = []
    for member in range(ensemble_size):
        print(f"Generating member {member+1}/{ensemble_size}")
        y_pred.append(dgmr(x))
    gc.collect()
    
    y_pred = np.stack(y_pred, axis=-1)
    return y_pred
