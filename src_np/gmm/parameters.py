"""Conversions between mixture weights and kernel-moment amplitudes."""

import numpy as np


def weights_from_amplitudes(means, sigmas, amplitudes, s):
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
    dimension = means.shape[1]
    weights = amplitudes * np.exp(
        0.5 * dimension * np.log1p((sigmas * sigmas) / (float(s) ** 2))
    )
    total = np.sum(weights)
    if total <= np.finfo(float).tiny:
        return np.full_like(weights, 1.0 / len(weights), dtype=float)
    return weights / total


def amplitudes_from_weights(means, sigmas, weights, s):
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    dimension = means.shape[1]
    return weights * np.exp(
        -0.5 * dimension * np.log1p((sigmas * sigmas) / (float(s) ** 2))
    )


def mixture_params_dict(means, sigmas, weights):
    return {
        "means": means,
        "weights": weights,
        "covariances": sigmas * sigmas,
    }
