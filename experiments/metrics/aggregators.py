from abc import ABC, abstractmethod
from math import sqrt
from statistics import median
from typing import TypeAlias


RebuiltMetrics: TypeAlias = dict[str, dict[str, list[float]]]
AggregatedMetrics: TypeAlias = dict[str, dict[str, float]]


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list of values")

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    weight = position - lower_index

    return (
        sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    )


class BaseMetricAggregator(ABC):
    def __init__(self, name):
        self.name = name

    def __call__(self, data: RebuiltMetrics) -> AggregatedMetrics:
        self._model_names = list(data.keys())

        result: AggregatedMetrics = {}

        for model_name, metrics in data.items():
            result[model_name] = {}

            for metric_name, values in metrics.items():
                if not values:
                    raise ValueError(
                        f"Cannot aggregate empty values for "
                        f"model={model_name!r}, metric={metric_name!r}"
                    )

                result[model_name][metric_name] = self._aggregate(values)

        return result

    def model_names(self) -> list[str]:
        if not hasattr(self, "_model_names"):
            raise RuntimeError("model_names is unavailable before __call__")

        return self._model_names.copy()

    @abstractmethod
    def _aggregate(self, values: list[float]) -> float:
        raise NotImplementedError


class MeanAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("mean")

    def _aggregate(self, values: list[float]) -> float:
        return sum(values) / len(values)


class StdAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("std")

    def _aggregate(self, values: list[float]) -> float:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)

        return sqrt(variance)


class MedianAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("median")

    def _aggregate(self, values: list[float]) -> float:
        return median(values)


class IQRAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("iqr")

    def _aggregate(self, values: list[float]) -> float:
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)

        return q3 - q1
