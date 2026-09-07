from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .declarations import DeclarationError
from .result_files import read_csv_rows, read_result_directory


GROUP_FIELDS = {
    "model": "model_name",
    "dataset": "dataset_name",
    "run": "run_name",
}

STATISTICS = {
    "mean_std": {
        "center_file": "mean",
        "spread_file": "std",
        "center_label": "mean",
        "spread_label": "std",
        "spread_half_width": 1.0,
    },
    "median_iqr": {
        "center_file": "median",
        "spread_file": "iqr",
        "center_label": "median",
        "spread_label": "IQR",
        "spread_half_width": 0.5,
    },
}


class ResultsCatalog:
    """Read and aggregate persisted experiment results from ``results/``."""

    def __init__(self, results_root: str | Path):
        self.results_root = Path(results_root).resolve()

    def saved_runs(self) -> list[dict[str, Any]]:
        if not self.results_root.exists():
            return []
        runs = []
        for run_path in sorted(self.results_root.iterdir()):
            config_path = run_path / "config.yaml"
            if not run_path.is_dir() or not config_path.is_file():
                continue
            dataset_names = []
            model_names = set()
            for mean_path in sorted(run_path.glob("*/mean.csv")):
                dataset_names.append(mean_path.parent.name)
                try:
                    rows = read_csv_rows(mean_path)
                except (OSError, UnicodeError, csv.Error):
                    continue
                model_names.update(
                    row["model_name"]
                    for row in rows
                    if isinstance(row.get("model_name"), str) and row["model_name"]
                )
            runs.append(
                self._saved_run_record(
                    run_path,
                    config=None,
                    dataset_names=dataset_names,
                    model_names=sorted(model_names, key=str.casefold),
                )
            )
        return sorted(runs, key=lambda run: run["saved_at"], reverse=True)

    def saved_run(self, run_name: str) -> dict[str, Any]:
        run_path = (self.results_root / run_name).resolve()
        if (
            run_path.parent != self.results_root
            or not run_path.is_dir()
            or not (run_path / "config.yaml").is_file()
        ):
            raise SavedRunNotFound(run_name)
        try:
            config = self._load_config(run_path / "config.yaml")
        except (OSError, ValueError, OmegaConfBaseException) as exc:
            raise DeclarationError(
                f"Could not read saved run configuration {run_name!r}: {exc}"
            ) from exc
        run = self._saved_run_record(run_path, config=config)
        return {
            "run": run,
            "results": {
                "run_id": run_name,
                "status": "saved",
                "result_path": str(run_path),
                "datasets": read_result_directory(run_path),
            },
        }

    def compare(
        self,
        *,
        run: str | None = None,
        dataset: str | None = None,
        model: str | None = None,
        group_by: str = "model",
        statistic: str = "mean_std",
    ) -> dict[str, Any]:
        if group_by not in GROUP_FIELDS:
            raise DeclarationError(
                f"group_by must be one of {sorted(GROUP_FIELDS)!r}"
            )
        if statistic not in STATISTICS:
            raise DeclarationError(
                f"statistic must be one of {sorted(STATISTICS)!r}"
            )

        statistic_config = STATISTICS[statistic]
        records, metric_names, warnings = self._scan(statistic_config)
        available = {
            "runs": self._unique(records, "run_name"),
            "datasets": self._unique(records, "dataset_name"),
            "models": self._unique(records, "model_name"),
        }
        selected = [
            record
            for record in records
            if (run is None or record["run_name"] == run)
            and (dataset is None or record["dataset_name"] == dataset)
            and (model is None or record["model_name"] == model)
        ]

        group_field = GROUP_FIELDS[group_by]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in selected:
            grouped[record[group_field]].append(record)

        metrics = []
        for metric_name in metric_names:
            values = []
            for group_name in sorted(grouped, key=str.casefold):
                observations = [
                    record["metrics"][metric_name]["value"]
                    for record in grouped[group_name]
                    if metric_name in record["metrics"]
                ]
                if not observations:
                    continue
                spreads = [
                    record["metrics"][metric_name]["spread"]
                    for record in grouped[group_name]
                    if metric_name in record["metrics"]
                    and record["metrics"][metric_name]["spread"] is not None
                ]
                values.append(
                    {
                        "name": group_name,
                        "value": math.fsum(observations) / len(observations),
                        "spread": (
                            math.fsum(spreads) / len(spreads) if spreads else None
                        ),
                        "observations": len(observations),
                        "spread_observations": len(spreads),
                    }
                )
            if values:
                metrics.append({"name": metric_name, "values": values})

        return {
            "source": str(self.results_root),
            "available": available,
            "selection": {
                "run": run,
                "dataset": dataset,
                "model": model,
            },
            "group_by": group_by,
            "statistic": statistic,
            "center_label": statistic_config["center_label"],
            "spread_label": statistic_config["spread_label"],
            "spread_half_width": statistic_config["spread_half_width"],
            "total_records": len(records),
            "matched_records": len(selected),
            "metrics": metrics,
            "warnings": warnings,
        }

    def _scan(
        self,
        statistic_config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        records = []
        metric_names = []
        known_metrics = set()
        warnings = []
        if not self.results_root.exists():
            return records, metric_names, warnings

        center_file = statistic_config["center_file"]
        spread_file = statistic_config["spread_file"]
        for center_path in sorted(self.results_root.glob(f"*/*/{center_file}.csv")):
            run_name = center_path.parent.parent.name
            dataset_name = center_path.parent.name
            try:
                rows = read_csv_rows(center_path)
            except (OSError, UnicodeError, csv.Error) as exc:
                warnings.append(f"Could not read {center_path}: {exc}")
                continue

            spread_rows = self._spread_rows(
                center_path.parent / f"{spread_file}.csv", warnings
            )

            for row_number, row in enumerate(rows, start=2):
                model_name = row.get("model_name")
                if not isinstance(model_name, str) or not model_name:
                    warnings.append(
                        f"Skipped {center_path}:{row_number}: model_name is missing"
                    )
                    continue
                metrics = {}
                for name, value in row.items():
                    if not self._is_metric(name, value):
                        continue
                    spread = spread_rows.get(model_name, {}).get(name)
                    if not self._is_finite_number(spread) or spread < 0:
                        spread = None
                    metrics[name] = {
                        "value": float(value),
                        "spread": float(spread) if spread is not None else None,
                    }
                    if name not in known_metrics:
                        known_metrics.add(name)
                        metric_names.append(name)
                records.append(
                    {
                        "run_name": run_name,
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "metrics": metrics,
                    }
                )
        return records, metric_names, warnings

    @classmethod
    def _spread_rows(
        cls,
        path: Path,
        warnings: list[str],
    ) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            rows = read_csv_rows(path)
        except (OSError, UnicodeError, csv.Error) as exc:
            warnings.append(f"Could not read {path}: {exc}")
            return {}
        return {
            row["model_name"]: row
            for row in rows
            if isinstance(row.get("model_name"), str) and row["model_name"]
        }

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    @classmethod
    def _is_metric(cls, name: Any, value: Any) -> bool:
        return (
            isinstance(name, str)
            and name != "model_name"
            and bool(name)
            and not name.startswith("Unnamed:")
            and cls._is_finite_number(value)
        )

    @staticmethod
    def _unique(records: list[dict[str, Any]], key: str) -> list[str]:
        return sorted({record[key] for record in records}, key=str.casefold)

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        config = OmegaConf.load(config_path)
        value = OmegaConf.to_container(config, resolve=True)
        if not isinstance(value, dict):
            raise ValueError("config.yaml must contain an object")
        return value

    @staticmethod
    def _saved_run_record(
        run_path: Path,
        config: dict[str, Any] | None,
        *,
        dataset_names: list[str] | None = None,
        model_names: list[str] | None = None,
    ) -> dict[str, Any]:
        if config is not None:
            datasets = config.get("datasets", [])
            models = config.get("models", [])
            dataset_names = [
                str(item.get("name", "unnamed dataset"))
                for item in datasets
                if isinstance(item, dict)
            ]
            model_names = [
                str(item.get("model_name", "unnamed model"))
                for item in models
                if isinstance(item, dict)
            ]
        dataset_names = dataset_names or []
        model_names = model_names or []
        config_path = run_path / "config.yaml"
        saved_at = max(run_path.stat().st_mtime, config_path.stat().st_mtime)
        record = {
            "id": run_path.name,
            "source": "saved",
            "status": "saved",
            "created_at": saved_at,
            "started_at": None,
            "finished_at": saved_at,
            "saved_at": saved_at,
            "return_code": None,
            "error": None,
            "metadata": {
                "save_result": True,
                "run_name": str(
                    config.get("run_name", run_path.name)
                    if config is not None
                    else run_path.name
                ),
                "result_path": str(run_path),
                "model_names": model_names,
                "dataset_names": dataset_names,
            },
            "config_path": str(config_path),
            "log_path": None,
            "log_tail": (
                "Worker log is unavailable because this run was loaded from the "
                "results directory rather than the current UI server session."
            ),
        }
        if config is not None:
            record["config"] = config
        return record


class SavedRunNotFound(KeyError):
    pass
