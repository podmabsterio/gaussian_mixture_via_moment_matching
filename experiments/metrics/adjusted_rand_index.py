from experiments.metrics.base_metric import BaseMetric

from sklearn.metrics import adjusted_rand_score


class AdjustedRandIndex(BaseMetric):
    def __init__(self):
        super().__init__("ARI")

    def __call__(self, labels, true_labels, **kwargs):
        return float(adjusted_rand_score(true_labels, labels))
