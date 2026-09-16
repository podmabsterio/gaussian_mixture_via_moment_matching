import time
from pathlib import Path

import numpy as np
import pytest

from experiments.visualization import PCAProjector2D
from src_np.iteration import IterationSnapshot, MixtureParameters
from ui.backend.app import create_app
from ui.backend.declarations import DeclarationStore
from ui.backend.quick_experiment import QuickExperimentManager
from ui.backend.results_catalog import ResultsCatalog
from ui.backend.run_manager import RunManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def quick_request(run_name="quick_test"):
    return {
        "model": {
            "declaration_id": "moment_gmm",
            "instance_name": "quick_model",
            "parameters": {
                "s_values": [2.0],
                "target_neighbor_counts": None,
                "target_neighbor_fractions": None,
                "n_init": 1,
                "n_zero_moments": 1,
                "n_first_moments": 0,
                "n_second_moments": 0,
                "n_radial_second_moments": 0,
                "max_steps": 1,
            },
        },
        "dataset": {
            "declaration_id": "gaussian",
            "instance_name": "quick_dataset",
            "parameters": {
                "n_features": 2,
                "n_components": 2,
                "n_samples": 40,
                "covariance_type": "spherical",
                "min_mahalanobis_distance": 3.0,
            },
        },
        "execution": {
            "run_name": run_name,
            "dataset_seed": 7,
            "model_seed": 11,
            "metric_interval": 1,
            "threads_limit": 1,
        },
    }


def test_quick_run_streams_iterations_and_saves_compatible_results(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    run_manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    catalog = ResultsCatalog(tmp_path / "results")
    quick_manager = QuickExperimentManager(
        store,
        PROJECT_ROOT,
        tmp_path / "runtime" / "quick",
        catalog.results_root,
        max_points=100,
    )
    client = create_app(store, run_manager, catalog, quick_manager).test_client()

    response = client.post("/api/quick-runs", json=quick_request())
    assert response.status_code == 202
    run_id = response.get_json()["id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        run = client.get(f"/api/quick-runs/{run_id}").get_json()
        if run["status"] not in {"starting", "running", "cancelling"}:
            break
        time.sleep(0.02)

    assert run["status"] == "completed", run.get("traceback") or run.get("error")
    assert run["data"]["total_points"] == run["data"]["displayed_points"] == 40
    assert len(run["data"]["true_components"]) == 2
    assert [snapshot["iteration"] for snapshot in run["snapshots"]] == [0, 1]
    assert all(
        len(snapshot["estimated_components"]) == 2 for snapshot in run["snapshots"]
    )
    assert set(run["snapshots"][0]["metrics"]) == {
        metric["id"] for metric in store.metrics
    }
    assert len(run["final_metrics"]) == len(store.metrics)
    assert all(metric["value"] is not None for metric in run["final_metrics"])

    events, terminal = quick_manager.events_since(run_id, 0, timeout=0)
    assert terminal is True
    assert {event["type"] for event in events} >= {
        "data",
        "iteration",
        "complete",
    }
    stream = client.get(f"/api/quick-runs/{run_id}/events?after=0")
    assert stream.mimetype == "text/event-stream"
    assert b'"type": "complete"' in stream.data

    saved = client.post(
        f"/api/quick-runs/{run_id}/save",
        json={"run_name": "saved_quick"},
    )
    assert saved.status_code == 200
    result_path = tmp_path / "results" / "saved_quick"
    assert (result_path / "config.yaml").is_file()
    assert (result_path / "quick_trace.json").is_file()
    assert (result_path / "quick_dataset" / "mean.csv").is_file()
    assert (result_path / "quick_dataset" / "std.csv").is_file()
    assert [item["id"] for item in catalog.saved_runs()] == ["saved_quick"]


def test_quick_data_point_initialization_uses_one_component_per_sample(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    run_manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    catalog = ResultsCatalog(tmp_path / "results")
    quick_manager = QuickExperimentManager(
        store,
        PROJECT_ROOT,
        tmp_path / "runtime" / "quick",
        catalog.results_root,
        max_points=100,
    )
    client = create_app(store, run_manager, catalog, quick_manager).test_client()
    request = quick_request("data_point_components")
    request["model"]["parameters"].update(
        {
            "init": "data_points",
            "max_steps": 0,
        }
    )
    request["dataset"]["parameters"]["n_samples"] = 20

    response = client.post("/api/quick-runs", json=request)
    assert response.status_code == 202
    run_id = response.get_json()["id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        run = client.get(f"/api/quick-runs/{run_id}").get_json()
        if run["status"] not in {"starting", "running", "cancelling"}:
            break
        time.sleep(0.02)

    assert run["status"] == "completed", run.get("traceback") or run.get("error")
    assert run["config"]["models"][0]["target"]["n_components"] == 20
    assert len(run["data"]["true_components"]) == 2
    assert len(run["snapshots"]) == 1
    displayed_components = run["snapshots"][0]["estimated_components"]
    assert 0 < len(displayed_components) <= 20
    assert all(component["weight"] > 1e-5 for component in displayed_components)
    assert all(metric["value"] is not None for metric in run["final_metrics"])


@pytest.mark.parametrize("mode", ["homogeneous", "inhomogeneous"])
def test_smooth_em_quick_run_streams_snapshots_and_metrics(tmp_path, mode):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    run_manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    catalog = ResultsCatalog(tmp_path / "results")
    quick_manager = QuickExperimentManager(
        store, PROJECT_ROOT, tmp_path / "runtime" / "quick", catalog.results_root
    )
    client = create_app(store, run_manager, catalog, quick_manager).test_client()
    request = quick_request(f"smooth_em_{mode}")
    request["model"] = {
        "declaration_id": "smooth_em_gmm",
        "instance_name": "smooth_em",
        "parameters": {
            "mode": mode,
            "n_design_points": 12,
            "initial_components": 6,
            "max_steps": 1,
            "internal_steps": 1,
            "weight_steps": 8,
        },
    }

    response = client.post("/api/quick-runs", json=request)
    assert response.status_code == 202
    run_id = response.get_json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        run = client.get(f"/api/quick-runs/{run_id}").get_json()
        if run["status"] not in {"starting", "running", "cancelling"}:
            break
        time.sleep(0.02)

    assert run["status"] == "completed", run.get("traceback") or run.get("error")
    assert "n_components" not in run["config"]["models"][0]["target"]
    assert len(run["snapshots"]) > 1
    assert run["snapshots"][0]["iteration"] == 0
    assert all(metric["value"] is not None for metric in run["final_metrics"])


def test_quick_snapshot_hides_negligible_estimated_components():
    means = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    weights = np.array([0.8, 1e-6, 0.2 - 1e-6])
    covariances = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    parameters = MixtureParameters(means, weights, covariances)
    snapshot = IterationSnapshot(0, 1.0, parameters)
    projector = PCAProjector2D(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))

    payload = QuickExperimentManager._snapshot_payload(
        snapshot,
        parameters.as_dict(),
        projector,
        {},
        {},
    )

    assert [component["index"] for component in payload["estimated_components"]] == [
        0,
        2,
    ]


def test_quick_run_validates_names_and_unknown_ids(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    run_manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    catalog = ResultsCatalog(tmp_path / "results")
    client = create_app(store, run_manager, catalog).test_client()

    unsafe = quick_request("../outside")
    assert client.post("/api/quick-runs", json=unsafe).status_code == 400

    unknown = quick_request()
    unknown["model"]["declaration_id"] = "missing"
    assert client.post("/api/quick-runs", json=unknown).status_code == 400
    assert client.get("/api/quick-runs/missing").status_code == 404
