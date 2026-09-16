import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from ui.backend.config_compiler import ConfigCompiler
from ui.backend.declarations import DeclarationStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compiled_config_runs_existing_pipeline(tmp_path):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path / "runtime")
    request = {
        "models": [
            {
                "declaration_id": "moment_gmm",
                "instance_name": "smoke_model",
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
            }
        ],
        "datasets": [
            {
                "declaration_id": "gaussian",
                "instance_name": "smoke_dataset",
                "parameters": {
                    "n_features": 2,
                    "n_components": 2,
                    "n_samples": 40,
                    "min_mahalanobis_distance": 3.0,
                },
            }
        ],
        "execution": {
            "save_result": False,
            "num_datasets": 1,
            "model_seeds_per_dataset": 1,
            "n_jobs": 1,
        },
    }
    config, metadata = compiler.compile(request, "smoke_test")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)

    completed = subprocess.run(
        [sys.executable, "-m", "ui.backend.worker", str(config_path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout
    result_path = Path(metadata["result_path"])
    assert (result_path / "config.yaml").exists()
    assert (result_path / "smoke_dataset" / "mean.csv").exists()


@pytest.mark.parametrize("mode", ["homogeneous", "inhomogeneous"])
def test_smooth_em_compiled_config_runs_and_saves_successful_fit(tmp_path, mode):
    store = DeclarationStore(PROJECT_ROOT / "ui" / "declarations")
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path / "runtime")
    request = {
        "models": [{
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
        }],
        "datasets": [{
            "declaration_id": "gaussian",
            "instance_name": "gaussian",
            "parameters": {
                "n_features": 2,
                "n_components": 2,
                "n_samples": 40,
                "min_mahalanobis_distance": 3.0,
            },
        }],
        "execution": {
            "save_result": False,
            "save_raw_results": True,
            "num_datasets": 1,
            "model_seeds_per_dataset": 1,
            "n_jobs": 1,
        },
    }
    config, metadata = compiler.compile(request, f"smooth_em_{mode}")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)

    completed = subprocess.run(
        [sys.executable, "-m", "ui.backend.worker", str(config_path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout
    result_path = Path(metadata["result_path"]) / "gaussian"
    raw = pd.read_csv(result_path / "raw.csv")
    assert raw.loc[0, "model_name"] == "smooth_em"
    assert raw.loc[0, "ok"] == 1
    assert (result_path / "mean.csv").exists()
