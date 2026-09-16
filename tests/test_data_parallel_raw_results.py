import pandas as pd
from omegaconf import OmegaConf

from experiments.runners.data_parallel_runner import _init_model, raw_results_dataframe


def test_raw_results_dataframe_preserves_paired_seed_identifiers():
    seeded_results = [
        (
            101,
            {
                "model_a": [{"ARI": 0.4, "ok": 1}, {"ARI": 0.5, "ok": 1}],
                "model_b": [{"ARI": 0.6, "ok": 1}, {"ARI": 0.7, "ok": 1}],
            },
        ),
        (
            102,
            {
                "model_a": [{"ARI": 0.8, "ok": 1}, {"ARI": 0.9, "ok": 1}],
                "model_b": [{"ARI": 1.0, "ok": 1}, {"ARI": 0.3, "ok": 1}],
            },
        ),
    ]

    raw = raw_results_dataframe(seeded_results)

    assert isinstance(raw, pd.DataFrame)
    assert len(raw) == 8
    assert set(raw["dataset_seed"]) == {101, 102}
    assert set(raw["model_seed"]) == {1, 2}
    assert set(raw["model_name"]) == {"model_a", "model_b"}
    assert raw.duplicated(["dataset_seed", "model_seed", "model_name"]).sum() == 0


def test_runner_respects_data_point_component_count():
    model_cfg = OmegaConf.create(
        {"target": {"_target_": "src_np.gmm.MomentGaussianMixtureModel", "init": "data_points"}}
    )
    dataset_cfg = OmegaConf.create({"n_components": 3})
    model = _init_model(model_cfg, 7, dataset_cfg, True, 20)

    assert model.k == 20


def test_runner_does_not_inject_component_count_into_smooth_em():
    model_cfg = OmegaConf.create(
        {"target": {"_target_": "src_np.gmm.SmoothEMGaussianMixtureModel"}}
    )
    dataset_cfg = OmegaConf.create({"n_components": 3})
    model = _init_model(model_cfg, 7, dataset_cfg, True, 20)

    assert model.initial_components is None
    assert model.random_state == 7
