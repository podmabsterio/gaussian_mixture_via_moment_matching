import json

import numpy as np

from src_np.gmm import SmoothEMGaussianMixtureModel
from src_np.iteration import IterationSnapshot


def _data(seed=7):
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.55, size=(24, 2)),
            rng.normal([2.0, 0.3], 0.8, size=(26, 2)),
        ]
    )


def test_smooth_em_homogeneous_fulfills_model_contract_and_callback():
    snapshots = []
    model = SmoothEMGaussianMixtureModel(
        mode="homogeneous",
        n_design_points=30,
        max_steps=2,
        random_state=11,
    ).fit(_data(), iteration_callback=snapshots.append)

    params = model.params_dict()
    assert params["means"].ndim == 2
    assert params["weights"].shape == (model.means_.shape[0],)
    assert params["covariances"].shape == (model.means_.shape[0], 2, 2)
    np.testing.assert_allclose(np.sum(params["weights"]), 1.0)
    np.testing.assert_allclose(params["covariances"][:, 0, 0], 1.0)
    np.testing.assert_allclose(params["covariances"][:, 1, 1], 1.0)
    assert snapshots[0].phase == "initialization"
    assert all(isinstance(snapshot, IterationSnapshot) for snapshot in snapshots)
    assert all(snapshot.parameters.means.flags.writeable is False for snapshot in snapshots)


def test_smooth_em_inhomogeneous_has_bounded_spherical_variances_and_is_reproducible():
    X = _data(12)
    kwargs = dict(
        mode="inhomogeneous",
        n_design_points=32,
        max_steps=2,
        internal_steps=2,
        n_directions=4,
        random_state=19,
    )
    first = SmoothEMGaussianMixtureModel(**kwargs).fit(X)
    second = SmoothEMGaussianMixtureModel(**kwargs).fit(X)

    np.testing.assert_allclose(first.means_, second.means_)
    np.testing.assert_allclose(first.sigmas_, second.sigmas_)
    np.testing.assert_allclose(first.weights_, second.weights_)
    assert np.all(first.sigmas_ ** 2 >= first.min_variance)
    assert np.all(first.sigmas_ ** 2 <= first.max_variance)
    assert np.all(first.weights_ >= 0)
    np.testing.assert_allclose(np.sum(first.weights_), 1.0)


def test_smooth_em_defaults_to_one_initial_component_per_design_point():
    model = SmoothEMGaussianMixtureModel(
        mode="inhomogeneous",
        n_design_points=12,
        max_steps=0,
        random_state=23,
    ).fit(_data())

    assert model.history_[0].metadata["n_components"] == 12


def test_smooth_em_is_registered_in_ui_declarations():
    with open("ui/declarations/models_declaration.json", encoding="utf-8") as handle:
        models = json.load(handle)["models"]
    declaration = next(model for model in models if model["id"] == "smooth_em_gmm")
    assert declaration["target"] == "src_np.gmm.SmoothEMGaussianMixtureModel"
    assert {parameter["key"] for parameter in declaration["parameters"]} >= {
        "mode",
        "max_steps",
        "random_state",
    }
