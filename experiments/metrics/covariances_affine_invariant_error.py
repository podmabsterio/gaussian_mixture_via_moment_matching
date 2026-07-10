from experiments.metrics.base_metric import BaseMetric
from experiments.metrics.metric_utils import EPS

import numpy as np


class CovariancesAffineInvariantError(BaseMetric):
    def __init__(self):
        super().__init__("Covariance Error")

    def __call__(self, covariances, true_covariances, true_weights, **kwargs):
        value = 0.0

        for k in range(len(true_weights)):
            true_eigvals, true_eigvecs = np.linalg.eigh(true_covariances[k])
            true_eigvals = np.maximum(true_eigvals, EPS)

            inv_sqrt = (
                true_eigvecs @ np.diag(1.0 / np.sqrt(true_eigvals)) @ true_eigvecs.T
            )
            relative = inv_sqrt @ covariances[k] @ inv_sqrt

            relative_eigvals = np.linalg.eigvalsh(relative)
            relative_eigvals = np.maximum(relative_eigvals, EPS)

            value += true_weights[k] * np.mean(np.log(relative_eigvals) ** 2)

        return float(np.sqrt(value))
