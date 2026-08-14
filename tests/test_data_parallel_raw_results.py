import pandas as pd

from experiments.runners.data_parallel_runner import raw_results_dataframe


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
