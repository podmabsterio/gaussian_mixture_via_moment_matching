from pathlib import Path

from ui.backend.app import create_app
from ui.backend.declarations import DeclarationStore
from ui.backend.results_catalog import ResultsCatalog
from ui.backend.run_manager import RunManager


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_health_and_declarations_api(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    app = create_app(store, manager)
    client = app.test_client()

    assert client.get("/api/health").get_json() == {"status": "ok"}
    assert client.get("/quick").status_code == 200
    response = client.get("/api/declarations")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["models"]) == 1
    assert body["models"][0]["id"] == "moment_gmm"


def test_invalid_run_is_reported_as_json(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    app = create_app(store, manager)
    response = app.test_client().post("/api/runs", json={})
    assert response.status_code == 400
    assert "model" in response.get_json()["error"].lower()


def test_comparison_api_reads_persisted_results_not_session_runs(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    mean_path = tmp_path / "results" / "saved_run" / "gaussian" / "mean.csv"
    mean_path.parent.mkdir(parents=True)
    mean_path.write_text("model_name,Metric A\nmodel_a,2.5\n", encoding="utf-8")
    (mean_path.parent / "std.csv").write_text(
        "model_name,Metric A\nmodel_a,0.25\n", encoding="utf-8"
    )
    (mean_path.parent.parent / "config.yaml").write_text(
        """run_name: saved_run
models:
  - model_name: model_a
    target:
      _target_: example.Model
datasets:
  - name: gaussian
    target:
      _target_: example.Dataset
""",
        encoding="utf-8",
    )
    catalog = ResultsCatalog(tmp_path / "results")
    client = create_app(store, manager, catalog).test_client()

    response = client.get("/api/comparison?dataset=gaussian&group_by=model")

    assert response.status_code == 200
    body = response.get_json()
    assert body["available"]["runs"] == ["saved_run"]
    assert body["matched_records"] == 1
    assert body["metrics"][0]["values"][0]["name"] == "model_a"
    assert body["metrics"][0]["values"][0]["spread"] == 0.25

    saved_runs = client.get("/api/saved-runs").get_json()
    assert [run["id"] for run in saved_runs] == ["saved_run"]
    saved_detail = client.get("/api/saved-runs/saved_run")
    assert saved_detail.status_code == 200
    assert saved_detail.get_json()["run"]["config"]["run_name"] == "saved_run"


def test_data_preview_uses_declared_generator_and_returns_projection(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    app = create_app(store, manager)
    response = app.test_client().post(
        "/api/data-preview",
        json={
            "dataset": {
                "declaration_id": "gaussian",
                "parameters": {
                    "n_features": 3,
                    "n_components": 2,
                    "n_samples": 60,
                    "covariance_type": "spherical",
                    "min_mahalanobis_distance": 3.0,
                },
            },
            "seed": 17,
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["dataset"]["declaration_id"] == "gaussian"
    assert body["seed"] == 17
    assert body["n_features"] == 3
    assert body["total_points"] == body["displayed_points"] == 60
    assert len(body["points"]) == 60
    assert len(body["components"]) == 2
    assert len(body["explained_variance_ratio"]) == 2
    assert all(len(component["center"]) == 2 for component in body["components"])
    assert all(len(component["radii"]) == 2 for component in body["components"])


def test_data_preview_rejects_unknown_target_and_hidden_override(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    manager = RunManager(PROJECT_ROOT, tmp_path / "runtime")
    client = create_app(store, manager).test_client()

    unknown = client.post(
        "/api/data-preview",
        json={"dataset": {"declaration_id": "not-declared", "parameters": {}}},
    )
    assert unknown.status_code == 400

    hidden_spec = dict(store.dataset("gaussian")["parameters"][0])
    hidden_spec.update({"key": "backend_only", "default": 1, "hidden": True})
    store.dataset("gaussian")["parameters"].append(hidden_spec)
    hidden = client.post(
        "/api/data-preview",
        json={
            "dataset": {
                "declaration_id": "gaussian",
                "parameters": {"backend_only": 2},
            }
        },
    )
    assert hidden.status_code == 400
    assert "hidden" in hidden.get_json()["error"].lower()
