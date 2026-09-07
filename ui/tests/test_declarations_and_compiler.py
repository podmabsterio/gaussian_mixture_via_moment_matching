from pathlib import Path

import pytest

from ui.backend.config_compiler import ConfigCompileError, ConfigCompiler
from ui.backend.declarations import DeclarationError, DeclarationStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATIONS = PROJECT_ROOT / "ui" / "declarations"


def base_request():
    return {
        "models": [
            {
                "declaration_id": "moment_gmm",
                "instance_name": "moment_test",
                "parameters": {"n_init": 2},
            }
        ],
        "datasets": [
            {
                "declaration_id": "gaussian",
                "instance_name": "gaussian_test",
                "parameters": {"n_samples": 100},
            }
        ],
        "execution": {"run_name": "compiler_test", "n_jobs": 2},
    }


def test_declarations_load_and_expand_dataset_parameter_sets():
    store = DeclarationStore(DECLARATIONS)
    assert [model["target"] for model in store.models] == [
        "src_np.gmm.MomentGaussianMixtureModel"
    ]
    assert len(store.datasets) == 6
    gaussian_keys = {parameter["key"] for parameter in store.dataset("gaussian")["parameters"]}
    assert {"n_features", "n_components", "n_samples"} <= gaussian_keys
    assert [metric["id"] for metric in store.metrics] == [
        "forward_kl",
        "weights_tv",
        "means_error",
        "covariance_error",
        "ari",
        "macro_f1",
    ]
    assert {parameter["key"] for parameter in store.quick_parameters} >= {
        "dataset_seed",
        "model_seed",
        "metric_interval",
    }


def test_advanced_and_hidden_default_to_false_when_omitted():
    store = DeclarationStore(DECLARATIONS)
    n_init = next(
        parameter for parameter in store.model("moment_gmm")["parameters"]
        if parameter["key"] == "n_init"
    )
    assert n_init.get("advanced", False) is False
    assert n_init.get("hidden", False) is False


def test_compiler_applies_ui_and_hidden_defaults(tmp_path):
    store = DeclarationStore(DECLARATIONS)
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path)
    config, metadata = compiler.compile(base_request(), "abc123")
    assert config.models[0].target._target_ == "src_np.gmm.MomentGaussianMixtureModel"
    assert config.models[0].target.n_init == 2
    assert config.models[0].target.n_first_moments == 3
    assert config.models[0].target.random_state is None
    assert config.datasets[0].target.n_samples == 100
    assert config.runner.n_jobs == 2
    assert metadata["result_path"].endswith("results/compiler_test")


def test_hidden_parameter_cannot_be_overridden(tmp_path):
    store = DeclarationStore(DECLARATIONS)
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path)
    request = base_request()
    request["models"][0]["parameters"]["random_state"] = 123
    with pytest.raises(DeclarationError, match="hidden"):
        compiler.compile(request, "abc123")


def test_unsafe_run_name_is_rejected(tmp_path):
    store = DeclarationStore(DECLARATIONS)
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path)
    request = base_request()
    request["execution"]["run_name"] = "../outside"
    with pytest.raises(ConfigCompileError, match="run_name"):
        compiler.compile(request, "abc123")


def test_conflicting_bandwidth_parameters_are_rejected(tmp_path):
    store = DeclarationStore(DECLARATIONS)
    compiler = ConfigCompiler(store, PROJECT_ROOT, tmp_path)
    request = base_request()
    request["models"][0]["parameters"]["s_values"] = [1.0]
    with pytest.raises(DeclarationError, match="conflicts"):
        compiler.compile(request, "abc123")
