from hydra.utils import instantiate

from experiments.metrics.metric_utils import (
    sort_estimated_params_by_means,
    predict_gm_labels,
)


class Evaluator:
    def __init__(self, metrics_cfg_list):
        self.metrics = []
        for metric_cfg in metrics_cfg_list:
            self.metrics.append(instantiate(metric_cfg))

    def prepare_estimated_params(self, X, true_means, estimated):
        means, weights, covariances = sort_estimated_params_by_means(
            true_means,
            estimated["means"],
            estimated["weights"],
            estimated["covariances"],
        )

        labels = predict_gm_labels(
            X,
            estimated["means"],
            estimated["weights"],
            estimated["covariances"],
        )

        return {
            "means": means,
            "weights": weights,
            "covariances": covariances,
            "labels": labels,
        }

    def create_empty_metrics(self):
        result = {}
        for metric in self.metrics:
            result[metric.name] = 0.0

        return result

    def __call__(self, dataset, estimated, model):
        kwargs = {"model": model}
        estimated = self.prepare_estimated_params(
            dataset["X"], dataset["true_means"], estimated
        )
        kwargs.update(estimated)
        kwargs.update(dataset)

        result = {}
        for metric in self.metrics:
            result[metric.name] = metric(**kwargs)

        return result
