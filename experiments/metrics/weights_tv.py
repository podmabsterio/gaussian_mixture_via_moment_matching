from experiments.metrics.base_metric import BaseMetric

import numpy as np


class WeightsTV(BaseMetric):
    def __init__(self):
        super().__init__("Weights TV")

    def __call__(self, weights, true_weights, **kwargs):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        true_weights = np.asarray(true_weights, dtype=float).reshape(-1)
        if weights.size < true_weights.size:
            raise ValueError("estimated weights cannot contain fewer matched components")
        matched_error = np.linalg.norm(
            weights[: true_weights.size] - true_weights,
            ord=1,
        )
        unmatched_mass = np.sum(np.abs(weights[true_weights.size :]))
        return float(0.5 * (matched_error + unmatched_mass))
