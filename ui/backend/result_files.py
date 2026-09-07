from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def coerce_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def read_result_directory(result_path: Path) -> list[dict[str, Any]]:
    datasets = []
    if not result_path.exists():
        return datasets
    for mean_path in sorted(result_path.glob("*/mean.csv")):
        dataset_path = mean_path.parent
        aggregates = {}
        for aggregate in ("mean", "std", "median", "iqr", "raw"):
            csv_path = dataset_path / f"{aggregate}.csv"
            if csv_path.exists():
                aggregates[aggregate] = read_csv_rows(csv_path)
        datasets.append({"name": dataset_path.name, "aggregates": aggregates})
    return datasets
