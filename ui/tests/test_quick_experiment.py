import time
from pathlib import Path

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
