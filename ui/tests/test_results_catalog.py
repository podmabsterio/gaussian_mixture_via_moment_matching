from pathlib import Path

import pytest

from ui.backend.results_catalog import ResultsCatalog, SavedRunNotFound


def write_aggregate(
    root: Path,
    run: str,
    dataset: str,
    aggregate: str,
    contents: str,
) -> None:
    path = root / run / dataset / f"{aggregate}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def make_catalog(tmp_path: Path) -> ResultsCatalog:
    root = tmp_path / "results"
    aggregates = {
        ("run_a", "gaussian"): {
            "mean": "model_name,Error,ARI\nmodel_a,1,\nmodel_b,3,0.7\n",
            "std": "model_name,Error,ARI\nmodel_a,0.1,\nmodel_b,0.3,0.07\n",
            "median": "model_name,Error,ARI\nmodel_a,0.8,\nmodel_b,2.8,0.75\n",
            "iqr": "model_name,Error,ARI\nmodel_a,0.4,\nmodel_b,0.6,0.08\n",
        },
        ("run_b", "gaussian"): {
            "mean": "model_name,Error,ARI\nmodel_a,5,0.9\n",
            "std": "model_name,Error,ARI\nmodel_a,0.5,0.09\n",
            "median": "model_name,Error,ARI\nmodel_a,4.5,0.92\n",
            "iqr": "model_name,Error,ARI\nmodel_a,1,0.1\n",
        },
        ("run_b", "student_t"): {
            "mean": "model_name,Error,Tail score\nmodel_b,7,4\n",
            "std": "model_name,Error,Tail score\nmodel_b,0.7,0.4\n",
            "median": "model_name,Error,Tail score\nmodel_b,6,3.5\n",
            "iqr": "model_name,Error,Tail score\nmodel_b,2,0.8\n",
        },
    }
    for (run, dataset), files in aggregates.items():
        for aggregate, contents in files.items():
            write_aggregate(root, run, dataset, aggregate, contents)
    return ResultsCatalog(root)


def metric(result, name):
    return next(item for item in result["metrics"] if item["name"] == name)


def test_catalog_scans_all_runs_and_aggregates_available_metrics(tmp_path):
    result = make_catalog(tmp_path).compare(group_by="model")

    assert result["total_records"] == result["matched_records"] == 4
    assert result["available"] == {
        "runs": ["run_a", "run_b"],
        "datasets": ["gaussian", "student_t"],
        "models": ["model_a", "model_b"],
    }
    assert metric(result, "Error")["values"] == [
        {
            "name": "model_a",
            "value": 3.0,
            "spread": 0.3,
            "observations": 2,
            "spread_observations": 2,
        },
        {
            "name": "model_b",
            "value": 5.0,
            "spread": 0.5,
            "observations": 2,
            "spread_observations": 2,
        },
    ]
    assert metric(result, "ARI")["values"] == [
        {
            "name": "model_a",
            "value": 0.9,
            "spread": 0.09,
            "observations": 1,
            "spread_observations": 1,
        },
        {
            "name": "model_b",
            "value": 0.7,
            "spread": 0.07,
            "observations": 1,
            "spread_observations": 1,
        },
    ]
    assert metric(result, "Tail score")["values"] == [
        {
            "name": "model_b",
            "value": 4.0,
            "spread": 0.4,
            "observations": 1,
            "spread_observations": 1,
        }
    ]


def test_catalog_filters_independently_and_can_group_by_run(tmp_path):
    result = make_catalog(tmp_path).compare(
        dataset="gaussian",
        model="model_a",
        group_by="run",
    )

    assert result["matched_records"] == 2
    assert metric(result, "Error")["values"] == [
        {
            "name": "run_a",
            "value": 1.0,
            "spread": 0.1,
            "observations": 1,
            "spread_observations": 1,
        },
        {
            "name": "run_b",
            "value": 5.0,
            "spread": 0.5,
            "observations": 1,
            "spread_observations": 1,
        },
    ]


def test_catalog_switches_to_median_and_iqr_pair(tmp_path):
    result = make_catalog(tmp_path).compare(
        group_by="model",
        statistic="median_iqr",
    )

    assert result["center_label"] == "median"
    assert result["spread_label"] == "IQR"
    assert result["spread_half_width"] == 0.5
    assert metric(result, "Error")["values"] == [
        {
            "name": "model_a",
            "value": 2.65,
            "spread": 0.7,
            "observations": 2,
            "spread_observations": 2,
        },
        {
            "name": "model_b",
            "value": 4.4,
            "spread": 1.3,
            "observations": 2,
            "spread_observations": 2,
        },
    ]


def test_catalog_ignores_non_finite_and_non_numeric_cells(tmp_path):
    root = tmp_path / "results"
    write_aggregate(
        root,
        "run",
        "dataset",
        "mean",
        "model_name,Finite,Missing,Infinite,Text\nmodel,2,,nan,nope\n",
    )

    result = ResultsCatalog(root).compare()

    assert result["metrics"] == [
        {
            "name": "Finite",
            "values": [
                {
                    "name": "model",
                    "value": 2.0,
                    "spread": None,
                    "observations": 1,
                    "spread_observations": 0,
                }
            ],
        }
    ]


def test_saved_runs_are_listed_and_loaded_from_results_directory(tmp_path):
    catalog = make_catalog(tmp_path)
    config_path = catalog.results_root / "run_a" / "config.yaml"
    config_path.write_text(
        """run_name: run_a
save_dir: results
models:
  - model_name: model_a
    target:
      _target_: example.Model
      n_init: 2
datasets:
  - name: gaussian
    target:
      _target_: example.Dataset
      n_samples: 100
runner:
  n_jobs: 1
""",
        encoding="utf-8",
    )

    runs = catalog.saved_runs()
    detail = catalog.saved_run("run_a")

    assert [run["id"] for run in runs] == ["run_a"]
    assert runs[0]["source"] == "saved"
    assert runs[0]["metadata"]["dataset_names"] == ["gaussian"]
    assert runs[0]["metadata"]["model_names"] == ["model_a", "model_b"]
    assert detail["run"]["config"]["models"][0]["target"]["n_init"] == 2
    assert detail["results"]["datasets"][0]["aggregates"]["mean"][0][
        "model_name"
    ] == "model_a"

    with pytest.raises(SavedRunNotFound):
        catalog.saved_run("../outside")
