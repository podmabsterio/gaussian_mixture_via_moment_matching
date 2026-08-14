from experiments.metrics.base_metric import BaseMetric


class FinalLoss(BaseMetric):
    def __init__(self):
        super().__init__("Final loss")

    def __call__(
        self,
        model,
        **kwargs,
    ):
        if hasattr(model, "objective_"):
            return float(model.objective_)
        return float(model.history_per_s[-1][-1])
