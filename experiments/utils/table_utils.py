import pandas as pd
from typing import List
from experiments.metrics.aggregators import BaseMetricAggregator


def results_to_dataframe(results: dict) -> pd.DataFrame:
    df = (
        pd.DataFrame.from_dict(results, orient="index")
        .rename_axis("model_name")
        .reset_index()
    )
    return df
