from experiments.metrics.base_metric import BaseMetric

from sklearn.metrics import f1_score


class MacroF1(BaseMetric):
    def __init__(self):
        super().__init__("Macro F1")

    def __call__(self, labels, true_labels, **kwargs):
        return float(f1_score(true_labels, labels, average="macro", zero_division=0))
