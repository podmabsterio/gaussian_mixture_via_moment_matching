from experiments.metrics.base_metric import BaseMetric


class FinalLoss(BaseMetric):
    def __init__(self):
        super().__init__("Final loss")

    def __call__(
        self,
        model,
        **kwargs,
    ):
        return float(model.history_per_s[-1][-1])
