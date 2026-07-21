from experiments.metrics.base_metric import BaseMetric
from experiments.metrics.metric_utils import log_gm_pdf
from experiments.synthetic_data import sample_gm_data

import numpy as np


class ForwardKL(BaseMetric):
    def __init__(self, num_samples=10000, seed=1):
        super().__init__("Forward KL")
        self.num_samples = num_samples
        self.seed = seed

    def __call__(
        self,
        means,
        weights,
        covariances,
        true_means,
        true_weights,
        true_covariances,
        **kwargs
    ):
        X, _ = sample_gm_data(
            true_means, true_weights, true_covariances, self.num_samples, seed=self.seed
        )

        if isinstance(X, tuple):
            X = X[0]

        true_log_pdf = log_gm_pdf(X, true_means, true_weights, true_covariances)
        estimated_log_pdf = log_gm_pdf(X, means, weights, covariances)

        return float(np.mean(true_log_pdf - estimated_log_pdf))
