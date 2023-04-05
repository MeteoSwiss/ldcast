import numpy as np


def transform_to_rainrate(x, mean=-0.051, std=0.528, threshold=0.1):
    x = x*std + mean
    R = 10**x
    R[R < threshold] = 0
    return R


def transform_from_rainrate(
    R, mean=-0.051, std=0.528,
    threshold=0.1, fill_value=0.02
):
    R = R.copy()
    R[R < threshold] = fill_value
    return (np.log10(R)-mean) / std
