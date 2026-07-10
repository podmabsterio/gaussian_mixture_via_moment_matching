from experiments.metrics.base_metric import BaseMetric

import numpy as np


class WeightsTV(BaseMetric):
    def __init__(self):
        super().__init__("Weights TV")

    def __call__(self, weights, true_weights, **kwargs):
        return float(0.5 * np.linalg.norm(weights - true_weights, ord=1))
