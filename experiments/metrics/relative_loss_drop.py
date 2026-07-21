from experiments.metrics.base_metric import BaseMetric


class RelativeLossDrop(BaseMetric):
    def __init__(self):
        super().__init__("Relative loss drop")

    def __call__(
        self,
        model,
        **kwargs,
    ):
        start = model.history_per_s[0][0]
        end = model.history_per_s[-1][-1]

        return (start - end) / start
