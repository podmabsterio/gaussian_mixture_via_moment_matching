from collections import defaultdict
from typing import Any


def data_parallel_metrics_rebuild(
    data: list[dict[str, list[dict[str, float]]]],
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for item in data:
        for model_name, metrics_list in item.items():
            for metrics in metrics_list:
                for metric_name, value in metrics.items():
                    result[model_name][metric_name].append(value)

    return {model_name: dict(metrics) for model_name, metrics in result.items()}
