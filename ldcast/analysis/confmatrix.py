import os
import concurrent
import multiprocessing

import netCDF4
import numpy as np
from scipy.integrate import trapezoid

from ..features.io import load_batch


def confusion_matrix(fc_frac, obs_frac, prob_threshold):
    N = np.prod(fc_frac.shape)
    fc_above = fc_frac > prob_threshold
    obs_above = obs_frac > prob_threshold
    tp = np.count_nonzero(fc_above & obs_above) / N
    fp = np.count_nonzero(fc_above & ~obs_above) / N
    fn = np.count_nonzero(~fc_above & obs_above) / N
    tn = 1.0 - tp - fp - fn
    return np.array(((tp, fn), (fp, tn)))


def confusion_matrix_thresholds(fc_frac, obs_frac, thresholds):
    N_threads = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(N_threads) as executor:
        futures = [
            executor.submit(confusion_matrix, fc_frac, obs_frac, t)
            for t in thresholds
        ]
        conf_matrix = [f.result() for f in futures]
    return np.stack(conf_matrix, axis=-1)


def confusion_matrix_thresholds_leadtime(fc_frac, obs_frac, thresholds):
    N_threads = multiprocessing.cpu_count()
    conf_matrix = []
    with concurrent.futures.ThreadPoolExecutor(N_threads) as executor:
        for lt in range(fc_frac.shape[2]):
            futures = [
                executor.submit(confusion_matrix,
                    fc_frac[...,lt,:,:], obs_frac[...,lt,:,:], t)
                for t in thresholds
            ]
            conf_matrix_lt = [f.result() for f in futures]
            conf_matrix_lt = np.stack(conf_matrix_lt, axis=-1)
            conf_matrix.append(conf_matrix_lt)

    return np.stack(conf_matrix, axis=-2)
    


def precision(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    precision = tp / (tp + fp)
    precision[np.isnan(precision)] = 1.0
    return precision


def recall(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return tp / (tp + fn)


def false_alarm_ratio(conf_matrix):
    return 1.0 - precision(conf_matrix)


def intersection_over_union(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return tp / (tp+fp+fn)


def equitable_threat_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    tp_rnd = (tp+fn) * (tp+fp) / (tp+fp+tn+fn)
    return (tp-tp_rnd) / (tp+fp+fn-tp_rnd)


def peirce_skill_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return (tp*tn - fn*fp) / ((tp+fn) * (fp+tn))


def heidke_skill_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return 2 * (tp*tn - fn*fp) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))


def roc_area_under_curve(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    auc = trapezoid(tpr[::-1], x=fpr[::-1])
    return auc


def pr_area_under_curve(conf_matrix):
    prec = precision(conf_matrix)
    rec = recall(conf_matrix)

    if (rec[-1] != 0) or (prec[-1] != 1):
        rec = np.hstack((rec, 0.0))
        prec = np.hstack((prec, 1.0))

    auc = trapezoid(prec[::-1], x=rec[::-1])
    return auc


def cost_loss_value(conf_matrix, cost, loss, p_clim):
    ((tp, fn), (fp, tn)) = conf_matrix

    E_c = min(cost, p_clim*loss)
    E_p = p_clim * cost
    E_f = (tp+fp)*cost + fn*loss
    #print(cost, loss, p_clim, E_c, E_p, E_f[len(E_f)//2]/E_p, (E_f[len(E_f)//2] - E_c) / (E_p - E_c))
    return (E_f - E_c) / (E_p - E_c)
