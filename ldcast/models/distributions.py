import numpy as np
import torch


def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def sample_from_standard_normal(mean, log_var, num=None):
    std = (0.5 * log_var).exp()
    shape = mean.shape
    if num is not None:
        # expand channel 1 to create several samples
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:,None,...]
        std = std[:,None,...]
    return mean + std * torch.randn(shape, device=mean.device)


def ensemble_nll_normal(ensemble, sample, epsilon=1e-5):
    mean = ensemble.mean(dim=1)
    var = ensemble.var(dim=1, unbiased=True) + epsilon
    logvar = var.log()

    diff = sample[:,None,...] - mean
    logtwopi = np.log(2*np.pi)
    nll = (logtwopi + logvar + diff.square() / var).mean()
    return nll
