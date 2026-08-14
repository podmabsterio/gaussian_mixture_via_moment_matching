from experiments.metrics.base_metric import BaseMetric


class RelativeLossDrop(BaseMetric):
    def __init__(self):
        super().__init__("Relative loss drop")

    def __call__(
        self,
        model,
        **kwargs,
    ):
        results = getattr(model, "optimization_results_", None)
        if results and "initial_objective" in results[-1]:
            start = float(results[-1]["initial_objective"])
            end = float(results[-1]["objective"])
            if start == 0:
                return 0.0
            return (start - end) / abs(start)

        start = model.history_per_s[0][0]
        end = model.history_per_s[-1][-1]

        return (start - end) / start
