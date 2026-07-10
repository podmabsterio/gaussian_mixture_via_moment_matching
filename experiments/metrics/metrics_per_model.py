from hydra.utils import instantiate

from experiments.metrics.aggregators import BaseMetricAggregator
from experiments.utils.table_utils import results_to_dataframe


class MetricsPerModelAggregator:
    def __init__(self, aggregators_cfg):
        self.aggregators = []
        for aggregator_cfg in aggregators_cfg:
            self.aggregators.append(instantiate(aggregator_cfg))

    def aggregate(self, results):
        aggregated_results = {}
        for aggregator in self.aggregators:
            aggregated_results[aggregator.name] = results_to_dataframe(
                aggregator(results)
            )

        return aggregated_results
