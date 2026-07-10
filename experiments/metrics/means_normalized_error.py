from experiments.metrics.base_metric import BaseMetric

import numpy as np


class MeansNormalizedError(BaseMetric):
    def __init__(self):
        super().__init__("Means Error")

    def __call__(self, means, true_means, true_covariances, true_weights, **kwargs):
        value = 0.0

        for k in range(len(true_weights)):
            diff = means[k] - true_means[k]
            scale = np.trace(true_covariances[k])
            value += true_weights[k] * np.dot(diff, diff) / scale

        return float(np.sqrt(value))
