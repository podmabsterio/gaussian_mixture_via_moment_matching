from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from experiments.metrics.evaluator import Evaluator
from experiments.utils.params_shape_utils import convert_and_check_params
from experiments.visualization import PCAProjector2D
from src_np.iteration import IterationSnapshot

from .config_compiler import SAFE_RUN_NAME, _validate_instance_name
from .data_preview import dataset_labels, display_indices, ellipse_geometry
from .declarations import DeclarationError, DeclarationStore


TERMINAL_STATUSES = {"completed", "cancelled", "error"}


class QuickRunNotFound(KeyError):
    pass


class QuickRunConflict(RuntimeError):
    pass


class QuickRunCancelled(RuntimeError):
    pass


class QuickMetricEvaluator:
    """Evaluate declared metrics while isolating a failure to one metric."""

    def __init__(self, declarations: list[dict[str, Any]]):
        self.declarations = declarations
        metric_configs = [
            {"_target_": item["target"], **copy.deepcopy(item.get("arguments", {}))}
            for item in declarations
        ]
        self.evaluator = Evaluator(OmegaConf.create(metric_configs))

    def evaluate(
        self,
        dataset: dict[str, Any],
        parameters: dict[str, np.ndarray],
        model: Any,
        *,
        live_only: bool,
    ) -> tuple[dict[str, float | None], dict[str, str], dict[str, np.ndarray]]:
        estimated = self.evaluator.prepare_estimated_params(
            dataset["X"],
            dataset["true_means"],
            parameters,
        )
        kwargs = {"model": model, **estimated, **dataset}
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for declaration, metric in zip(
            self.declarations,
            self.evaluator.metrics,
            strict=True,
        ):
            metric_id = declaration["id"]
            if live_only and not declaration.get("track_during_fit", True):
                continue
            try:
                value = float(metric(**kwargs))
                if not np.isfinite(value):
                    raise ValueError("returned a non-finite value")
                values[metric_id] = value
            except Exception as exc:  # one optional metric must not hide the run
                values[metric_id] = None
                errors[metric_id] = str(exc)
        return values, errors, estimated

    def result_rows(
        self,
        values: dict[str, float | None],
    ) -> tuple[dict[str, float], dict[str, float]]:
        centers: dict[str, float] = {}
        spreads: dict[str, float] = {}
        for declaration, metric in zip(
            self.declarations,
            self.evaluator.metrics,
            strict=True,
        ):
            value = values.get(declaration["id"])
            if value is not None:
                centers[metric.name] = float(value)
                spreads[metric.name] = 0.0
        return centers, spreads


class QuickExperimentManager:
    """Run and observe one model fit without changing the batch pipeline."""

    def __init__(
        self,
        declarations: DeclarationStore,
        project_root: str | Path,
        runtime_dir: str | Path,
        results_root: str | Path,
        *,
        max_points: int = 3_000,
    ):
        self.declarations = declarations
        self.project_root = Path(project_root).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.results_root = Path(results_root).resolve()
        self.max_points = int(max_points)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, dict[str, Any]] = {}
        self._condition = threading.Condition(threading.RLock())

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start(self, request: Any) -> dict[str, Any]:
        prepared = self._prepare_request(request)
        run_id = self.new_id()
        if not prepared["metadata"]["run_name"]:
            generated_name = time.strftime("quick_%Y%m%d_%H%M%S") + f"_{run_id[:6]}"
            prepared["metadata"]["run_name"] = generated_name
            prepared["config"]["run_name"] = generated_name
        now = time.time()
        record: dict[str, Any] = {
            "id": run_id,
            "status": "starting",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "traceback": None,
            "metadata": prepared["metadata"],
            "config": prepared["config"],
            "execution": prepared["execution"],
            "data": None,
            "snapshots": [],
            "final_metrics": None,
            "final_metric_errors": {},
            "fit_time": None,
            "saved_path": None,
            "_prepared": prepared,
            "_cancel_event": threading.Event(),
            "_events": [],
            "_saving": False,
        }
        with self._condition:
            self._runs[run_id] = record
            self._publish_locked(
                record, {"type": "status", "run": self._summary(record)}
            )

        worker = threading.Thread(
            target=self._execute,
            args=(run_id,),
            daemon=True,
            name=f"quick-experiment-{run_id}",
        )
        worker.start()
        return self.get(run_id)

    def _prepare_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise DeclarationError("Request body must be an object")
        model_entry = request.get("model")
        dataset_entry = request.get("dataset")
        if not isinstance(model_entry, dict):
            raise DeclarationError("model must be an object")
        if not isinstance(dataset_entry, dict):
            raise DeclarationError("dataset must be an object")

        model_declaration = self.declarations.model(model_entry.get("declaration_id"))
        dataset_declaration = self.declarations.dataset(
            dataset_entry.get("declaration_id")
        )
        model_name = _validate_instance_name(
            model_entry.get(
                "instance_name", model_declaration["default_instance_name"]
            ),
            "model.instance_name",
        )
        dataset_name = _validate_instance_name(
            dataset_entry.get(
                "instance_name", dataset_declaration["default_instance_name"]
            ),
            "dataset.instance_name",
        )
        model_parameters = self.declarations.resolve_parameters(
            model_declaration["parameters"],
            model_entry.get("parameters", {}),
            "model.parameters",
        )
        dataset_parameters = self.declarations.resolve_parameters(
            dataset_declaration["parameters"],
            dataset_entry.get("parameters", {}),
            "dataset.parameters",
        )
        execution = self.declarations.resolve_quick_execution(
            request.get("execution", {})
        )
        requested_name = execution["run_name"].strip()
        if requested_name and not SAFE_RUN_NAME.fullmatch(requested_name):
            raise DeclarationError(
                "run_name must start with an ASCII letter or digit and contain "
                "only letters, digits, '.', '_' or '-'"
            )

        dataset_target = {
            "_target_": dataset_declaration["target"],
            **dataset_parameters,
        }
        model_target = {
            "_target_": model_declaration["target"],
            **model_parameters,
            "random_state": execution["model_seed"],
        }
        metric_configs = [
            {"_target_": item["target"], **copy.deepcopy(item.get("arguments", {}))}
            for item in self.declarations.metrics
        ]
        config = {
            "run_name": requested_name,
            "save_dir": "results",
            "override": False,
            "quick_experiment": True,
            "models": [{"model_name": model_name, "target": model_target}],
            "datasets": [{"name": dataset_name, "target": dataset_target}],
            "runner": {
                "_target_": "experiments.runners.data_parallel_runner.DataParallelRunner",
                "_recursive_": False,
                "num_datasets": 1,
                "model_seeds_per_dataset": 1,
                "n_jobs": 1,
                "threads_limit": execution["threads_limit"],
                "backend": "threading",
                "batch_size": "auto",
                "dataset_seed_start": execution["dataset_seed"],
                "init_models_with_oracle_n_components": True,
                "save_raw_results": True,
            },
            "metrics": metric_configs,
            "aggregators": copy.deepcopy(
                self.declarations.fixed_config.get("aggregators", [])
            ),
            "save_data_visualizations": False,
            "num_visualization_samples": 0,
            "quick_execution": copy.deepcopy(execution),
        }
        return {
            "model_declaration": model_declaration,
            "dataset_declaration": dataset_declaration,
            "model_parameters": model_parameters,
            "dataset_parameters": dataset_parameters,
            "execution": execution,
            "config": config,
            "metadata": {
                "run_name": requested_name,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "model_display_name": model_declaration["display_name"],
                "dataset_display_name": dataset_declaration["display_name"],
            },
        }

    def _execute(self, run_id: str) -> None:
        with self._condition:
            record = self._record(run_id)
            record["status"] = "running"
            record["started_at"] = time.time()
            prepared = record["_prepared"]
            cancel_event = record["_cancel_event"]
            self._publish_locked(
                record, {"type": "status", "run": self._summary(record)}
            )

        try:
            if cancel_event.is_set():
                raise QuickRunCancelled("Run cancelled before data generation")
            dataset_generator = instantiate(
                OmegaConf.create(
                    {
                        "_target_": prepared["dataset_declaration"]["target"],
                        **prepared["dataset_parameters"],
                    }
                )
            )
            dataset = dataset_generator.generate(prepared["execution"]["dataset_seed"])
            projector = PCAProjector2D(dataset["X"])
            data_payload = self._data_payload(
                dataset,
                projector,
                prepared["execution"]["dataset_seed"],
            )
            with self._condition:
                record = self._record(run_id)
                record["data"] = data_payload
                self._publish_locked(record, {"type": "data", "data": data_payload})

            n_components = int(np.asarray(dataset["true_means"]).shape[0])
            model_configuration = {
                "_target_": prepared["model_declaration"]["target"],
                **prepared["model_parameters"],
                "n_components": n_components,
                "random_state": prepared["execution"]["model_seed"],
            }
            prepared["config"]["models"][0]["target"] = copy.deepcopy(
                model_configuration
            )
            with self._condition:
                record = self._record(run_id)
                record["config"] = copy.deepcopy(prepared["config"])
            model = instantiate(OmegaConf.create(model_configuration))
            evaluator = QuickMetricEvaluator(self.declarations.metrics)
            metric_interval = prepared["execution"]["metric_interval"]

            def on_iteration(snapshot: IterationSnapshot) -> None:
                if cancel_event.is_set():
                    raise QuickRunCancelled("Run cancelled by user")
                evaluate_metrics = snapshot.iteration % metric_interval == 0
                parameters = snapshot.parameters.as_dict()
                metrics: dict[str, float | None] = {}
                metric_errors: dict[str, str] = {}
                if evaluate_metrics:
                    metrics, metric_errors, ordered = evaluator.evaluate(
                        dataset,
                        parameters,
                        model,
                        live_only=True,
                    )
                else:
                    ordered = evaluator.evaluator.prepare_estimated_params(
                        dataset["X"],
                        dataset["true_means"],
                        parameters,
                    )
                payload = self._snapshot_payload(
                    snapshot,
                    ordered,
                    projector,
                    metrics,
                    metric_errors,
                )
                with self._condition:
                    current = self._record(run_id)
                    current["snapshots"].append(payload)
                    self._publish_locked(
                        current,
                        {"type": "iteration", "snapshot": payload},
                    )

            fit_start = time.perf_counter()
            with threadpool_limits(limits=prepared["execution"]["threads_limit"]):
                model.fit(**dataset, iteration_callback=on_iteration)
            fit_time = time.perf_counter() - fit_start
            if cancel_event.is_set():
                raise QuickRunCancelled("Run cancelled by user")

            final_parameters = convert_and_check_params(model.params_dict())
            final_metrics, final_errors, _ = evaluator.evaluate(
                dataset,
                final_parameters,
                model,
                live_only=False,
            )
            if cancel_event.is_set():
                raise QuickRunCancelled("Run cancelled by user")
            final_payload = [
                {
                    "id": declaration["id"],
                    "name": declaration["display_name"],
                    "description": declaration["description"],
                    "direction": declaration.get("direction", "neutral"),
                    "format": declaration.get("format", ".5g"),
                    "value": final_metrics.get(declaration["id"]),
                    "error": final_errors.get(declaration["id"]),
                }
                for declaration in self.declarations.metrics
            ]
            with self._condition:
                record = self._record(run_id)
                record["status"] = "completed"
                record["finished_at"] = time.time()
                record["fit_time"] = fit_time
                record["final_metrics"] = final_payload
                record["final_metric_errors"] = final_errors
                record["_metric_centers"], record["_metric_spreads"] = (
                    evaluator.result_rows(final_metrics)
                )
                self._publish_locked(
                    record,
                    {
                        "type": "complete",
                        "run": self._summary(record),
                        "final_metrics": final_payload,
                    },
                )
        except QuickRunCancelled as exc:
            with self._condition:
                record = self._record(run_id)
                record["status"] = "cancelled"
                record["finished_at"] = time.time()
                record["error"] = str(exc)
                self._publish_locked(
                    record, {"type": "status", "run": self._summary(record)}
                )
        except Exception as exc:
            with self._condition:
                record = self._record(run_id)
                record["status"] = "error"
                record["finished_at"] = time.time()
                record["error"] = str(exc) or exc.__class__.__name__
                record["traceback"] = traceback.format_exc()
                self._publish_locked(
                    record, {"type": "status", "run": self._summary(record)}
                )

    def _data_payload(
        self,
        dataset: dict[str, Any],
        projector: PCAProjector2D,
        seed: int,
    ) -> dict[str, Any]:
        X = np.asarray(dataset["X"], dtype=float)
        projected = projector.transform(X)
        labels = dataset_labels(dataset, X.shape[0])
        indices = display_indices(labels, seed, self.max_points)
        true_means = projector.transform(dataset["true_means"])
        true_covariances = projector.transform_covariances(dataset["true_covariances"])
        weights = np.asarray(dataset["true_weights"], dtype=float)
        components = ellipse_geometry(true_means, true_covariances)
        return {
            "seed": int(seed),
            "n_features": int(X.shape[1]),
            "total_points": int(X.shape[0]),
            "displayed_points": int(indices.size),
            "explained_variance_ratio": [
                float(value) for value in projector.explained_variance_ratio_
            ],
            "points": [
                {
                    "x": float(projected[index, 0]),
                    "y": float(projected[index, 1]),
                    "label": int(labels[index]),
                }
                for index in indices
            ],
            "true_components": [
                {"index": index, "weight": float(weights[index]), **component}
                for index, component in enumerate(components)
            ],
        }

    @staticmethod
    def _snapshot_payload(
        snapshot: IterationSnapshot,
        ordered: dict[str, np.ndarray],
        projector: PCAProjector2D,
        metrics: dict[str, float | None],
        metric_errors: dict[str, str],
    ) -> dict[str, Any]:
        means = projector.transform(ordered["means"])
        covariances = projector.transform_covariances(ordered["covariances"])
        geometry = ellipse_geometry(means, covariances)
        weights = np.asarray(ordered["weights"], dtype=float)
        return {
            "iteration": snapshot.iteration,
            "loss": snapshot.loss,
            "losses": dict(snapshot.losses),
            "phase": snapshot.phase,
            "metadata": dict(snapshot.metadata),
            "metrics": metrics,
            "metric_errors": metric_errors,
            "estimated_components": [
                {"index": index, "weight": float(weights[index]), **component}
                for index, component in enumerate(geometry)
            ],
        }

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._condition:
            record = self._record(run_id)
            if record["status"] in TERMINAL_STATUSES:
                return self._public(record)
            record["status"] = "cancelling"
            record["_cancel_event"].set()
            self._publish_locked(
                record, {"type": "status", "run": self._summary(record)}
            )
            return self._public(record)

    def save(self, run_id: str, run_name: str | None = None) -> dict[str, Any]:
        with self._condition:
            record = self._record(run_id)
            if record["status"] != "completed":
                raise QuickRunConflict("Only a completed quick run can be saved")
            if record["saved_path"]:
                return {
                    "run_name": Path(record["saved_path"]).name,
                    "result_path": record["saved_path"],
                }
            if record["_saving"]:
                raise QuickRunConflict("This quick run is already being saved")
            name = (
                run_name
                or record["execution"].get("run_name")
                or record["metadata"]["run_name"]
            ).strip()
            if not SAFE_RUN_NAME.fullmatch(name):
                raise DeclarationError(
                    "run_name must start with an ASCII letter or digit and contain "
                    "only letters, digits, '.', '_' or '-'"
                )
            result_path = self.results_root / name
            if result_path.exists():
                raise QuickRunConflict(f"Run directory already exists: {result_path}")
            record["_saving"] = True
            snapshot = self._public(record)
            metric_centers = copy.deepcopy(record["_metric_centers"])
            metric_spreads = copy.deepcopy(record["_metric_spreads"])

        temporary_path = self.results_root / f".quick-{run_id}-{uuid.uuid4().hex[:8]}"
        try:
            self.results_root.mkdir(parents=True, exist_ok=True)
            temporary_path.mkdir(parents=False, exist_ok=False)
            config = copy.deepcopy(snapshot["config"])
            config["run_name"] = name
            OmegaConf.save(OmegaConf.create(config), temporary_path / "config.yaml")
            dataset_name = snapshot["metadata"]["dataset_name"]
            model_name = snapshot["metadata"]["model_name"]
            dataset_path = temporary_path / dataset_name
            dataset_path.mkdir()
            center_row = {"model_name": model_name, **metric_centers}
            spread_row = {"model_name": model_name, **metric_spreads}
            self._write_csv(dataset_path / "mean.csv", center_row)
            self._write_csv(dataset_path / "median.csv", center_row)
            self._write_csv(dataset_path / "std.csv", spread_row)
            self._write_csv(dataset_path / "iqr.csv", spread_row)
            self._write_csv(
                dataset_path / "raw.csv",
                {
                    "dataset_seed": snapshot["execution"]["dataset_seed"],
                    "model_seed": snapshot["execution"]["model_seed"],
                    **center_row,
                    "fit_time": snapshot["fit_time"],
                    "ok": 1,
                },
            )
            trace = {
                "quick_experiment": True,
                "fit_time": snapshot["fit_time"],
                "execution": snapshot["execution"],
                "snapshots": snapshot["snapshots"],
                "final_metrics": snapshot["final_metrics"],
            }
            (temporary_path / "quick_trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.rename(temporary_path, result_path)
        except Exception:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
            with self._condition:
                self._record(run_id)["_saving"] = False
            raise

        with self._condition:
            record = self._record(run_id)
            record["_saving"] = False
            record["saved_path"] = str(result_path)
            self._publish_locked(
                record,
                {"type": "saved", "run_name": name, "result_path": str(result_path)},
            )
        return {"run_name": name, "result_path": str(result_path)}

    @staticmethod
    def _write_csv(path: Path, row: dict[str, Any]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._condition:
            return self._public(self._record(run_id))

    def events_since(
        self,
        run_id: str,
        after: int,
        timeout: float = 15.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        deadline = time.monotonic() + timeout
        with self._condition:
            record = self._record(run_id)
            while (
                len(record["_events"]) <= after
                and record["status"] not in TERMINAL_STATUSES
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
                record = self._record(run_id)
            events = copy.deepcopy(record["_events"][after:])
            return events, record["status"] in TERMINAL_STATUSES

    def _record(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise QuickRunNotFound(run_id) from exc

    def _publish_locked(self, record: dict[str, Any], event: dict[str, Any]) -> None:
        sequence = len(record["_events"]) + 1
        record["_events"].append({"sequence": sequence, **event})
        self._condition.notify_all()

    @staticmethod
    def _summary(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(record[key])
            for key in (
                "id",
                "status",
                "created_at",
                "started_at",
                "finished_at",
                "error",
                "metadata",
                "fit_time",
                "saved_path",
            )
        }

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(
            {key: value for key, value in record.items() if not key.startswith("_")}
        )


__all__ = [
    "QuickExperimentManager",
    "QuickRunConflict",
    "QuickRunNotFound",
]
