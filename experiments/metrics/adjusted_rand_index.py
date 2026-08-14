from experiments.metrics.base_metric import BaseMetric

import numpy as np
from sklearn.metrics import adjusted_rand_score


class AdjustedRandIndex(BaseMetric):
    def __init__(self):
        super().__init__("ARI")

    def __call__(
        self,
        labels,
        true_labels,
        evaluation_mask=None,
        **kwargs,
    ):
        labels = np.asarray(labels)
        true_labels = np.asarray(true_labels)
        if labels.shape != true_labels.shape:
            raise ValueError("labels and true_labels must have matching shapes")

        if evaluation_mask is not None:
            evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
            if evaluation_mask.shape != true_labels.shape:
                raise ValueError(
                    "evaluation_mask must have the same shape as true_labels"
                )
            if not np.any(evaluation_mask):
                raise ValueError("evaluation_mask must retain at least one sample")
            labels = labels[evaluation_mask]
            true_labels = true_labels[evaluation_mask]

        return float(adjusted_rand_score(true_labels, labels))
