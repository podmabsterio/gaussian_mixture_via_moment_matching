import numpy as np
import pytest

from experiments.metrics import (
    FormulaVarianceEtaStepBoundHitRate,
    FormulaVarianceInvalidUpdateRate,
    FormulaVarianceQClipRate,
    FormulaVarianceUpdates,
    JointVarianceSweeps,
    MaximumRelativeOuterObjectiveIncrease,
    OuterObjectiveIncreaseRate,
)
from src_np.gmm import MomentGaussianMixtureModel
from src_np.optimization.formula_variance import (
    variance_from_corrected_radial_moments,
)


def _small_mixture():
    rng = np.random.default_rng(401)
    data = np.vstack(
        [
            rng.normal([-1.0, -0.7, 0.2], 0.5, size=(28, 3)),
            rng.normal([0.9, 1.1, -0.1], 0.8, size=(32, 3)),
        ]
    )
    init = {
        "means": np.array([[-0.8, -0.6, 0.1], [0.8, 0.9, 0.0]]),
        "sigmas": np.array([0.7, 0.7]),
        "weights": np.array([0.5, 0.5]),
    }
    return data, init


def _recent_experiment_configuration(data, init, *, optimizer_mode, iteration_callback=None):
    return MomentGaussianMixtureModel(
        2,
        s_values=[1.4, 0.75],
        init=init,
        n_zero_moments=0,
        n_first_moments=2,
        n_second_moments=3,
        n_radial_second_moments=1,
        zero_moment_weight=0.0,
        first_moment_weight=0.125,
        second_moment_weight=0.5,
        radial_second_moment_weight=0.5,
        normalize_moment_losses=True,
        scale_normalize_moments=True,
        leave_one_out=True,
        amplitude_optimization="simplex",
        s_optimization="joint",
        geometry_optimization="component",
        optimizer_mode=optimizer_mode,
        max_steps=3,
        geom_sweeps=1,
        ls_max_nfev=8,
        objective_rtol=0.0,
        convergence_patience=99,
        random_state=23,
    ).fit(data, iteration_callback=iteration_callback)


def test_variance_formula_exactly_recovers_component_after_neighbor_subtraction():
    dimension = 5
    s = 1.3
    means = np.array(
        [
            [-0.7, 0.1, 0.4, -0.2, 0.0],
            [1.0, -0.5, 0.2, 0.6, -0.3],
            [0.2, 0.8, -0.9, 0.1, 0.5],
        ]
    )
    variances = np.array([0.36, 1.21, 0.64])
    weights = np.array([0.2, 0.5, 0.3])
    component = 1
    center = means[component]
    offset = means - center
    norm2 = np.einsum("kd,kd->k", offset, offset)
    totals = s * s + variances
    zero_contributions = weights * np.exp(
        -0.5 * dimension * np.log1p(variances / (s * s)) - 0.5 * norm2 / totals
    )
    radial_contributions = zero_contributions * (
        s * s * norm2 / (dimension * totals * totals) + variances / totals
    )
    neighbor_mask = np.arange(means.shape[0]) != component
    corrected_zero = np.sum(zero_contributions) - np.sum(
        zero_contributions[neighbor_mask]
    )
    corrected_radial = np.sum(radial_contributions) - np.sum(
        radial_contributions[neighbor_mask]
    )

    actual, raw_q, clipped_q = variance_from_corrected_radial_moments(
        corrected_zero,
        corrected_radial,
        s,
        q_bounds=(1e-6, 1.0 - 1e-6),
    )

    assert raw_q == pytest.approx(variances[component] / totals[component])
    assert clipped_q == pytest.approx(raw_q)
    assert actual == pytest.approx(variances[component], rel=1e-13, abs=1e-13)


def test_variance_formula_clips_empirical_ratio_inside_open_unit_interval():
    variance, raw_q, clipped_q = variance_from_corrected_radial_moments(
        0.2,
        0.3,
        2.0,
        q_bounds=(0.05, 0.9),
    )
    assert raw_q == pytest.approx(1.5)
    assert clipped_q == pytest.approx(0.9)
    assert variance == pytest.approx(36.0)


@pytest.mark.parametrize("optimizer_mode", ["formula_variance", "joint_variance"])
def test_separated_variance_models_fit_recent_experiment_configuration(
    optimizer_mode,
):
    data, init = _small_mixture()
    model = _recent_experiment_configuration(
        data,
        init,
        optimizer_mode=optimizer_mode,
    )
    result = model.optimization_results_[0]

    assert np.isfinite(model.objective_)
    assert np.all(np.isfinite(model.means_))
    assert np.all(np.isfinite(model.sigmas_))
    assert np.all(model.sigmas_ > 0)
    assert np.all(model.weights_ >= 0)
    np.testing.assert_allclose(np.sum(model.weights_), 1.0, atol=1e-14)
    assert result["variance_update_mode"] == (
        "formula" if optimizer_mode == "formula_variance" else "joint"
    )
    assert result["history"].shape == (3,)
    assert np.isfinite(result["final_mean_gradient_inf"])
    assert np.isfinite(result["final_eta_gradient_inf"])

    if optimizer_mode == "formula_variance":
        assert result["variance_formula_s"] == pytest.approx(0.75)
        assert result["variance_formula_updates"] == 3 * 2
        assert result["variance_joint_calls"] == 0
    else:
        assert result["variance_formula_updates"] == 0
        assert result["variance_joint_calls"] == 3
        objectives = np.r_[result["initial_objective"], result["history"]]
        assert np.all(np.diff(objectives) <= 2e-14)


@pytest.mark.parametrize("optimizer_mode", ["formula_variance", "joint_variance"])
def test_separated_variance_modes_report_canonical_iteration_snapshots(
    optimizer_mode,
):
    data, init = _small_mixture()
    snapshots = []
    _recent_experiment_configuration(
        data,
        init,
        optimizer_mode=optimizer_mode,
        iteration_callback=snapshots.append,
    )

    assert [snapshot.iteration for snapshot in snapshots] == [0, 1, 2, 3]
    assert snapshots[0].phase == "initialization"
    assert all(snapshot.parameters.means.shape == (2, 3) for snapshot in snapshots)
    assert all(
        snapshot.parameters.covariances.shape == (2, 3, 3)
        for snapshot in snapshots
    )
    assert all(snapshot.parameters.weights.shape == (2,) for snapshot in snapshots)
    assert snapshots[-1].metadata["bandwidth_mode"] == "joint"
    assert snapshots[-1].parameters.means.flags.writeable is False


def test_explicit_legacy_mode_is_identical_to_unchanged_default_optimizer():
    data, init = _small_mixture()
    common = {
        "n_components": 2,
        "s_values": [1.4, 0.75],
        "init": init,
        "n_zero_moments": 0,
        "n_first_moments": 2,
        "n_second_moments": 3,
        "n_radial_second_moments": 1,
        "first_moment_weight": 0.125,
        "second_moment_weight": 0.5,
        "radial_second_moment_weight": 0.5,
        "leave_one_out": True,
        "amplitude_optimization": "simplex",
        "s_optimization": "joint",
        "geometry_optimization": "component",
        "max_steps": 2,
        "ls_max_nfev": 7,
        "objective_rtol": 0.0,
        "convergence_patience": 99,
        "random_state": 19,
    }
    default = MomentGaussianMixtureModel(**common).fit(data)
    explicit = MomentGaussianMixtureModel(
        **common,
        optimizer_mode="legacy",
    ).fit(data)
    snapshots = []
    observed = MomentGaussianMixtureModel(**common).fit(
        data,
        iteration_callback=snapshots.append,
    )

    np.testing.assert_array_equal(default.test_directions_, explicit.test_directions_)
    np.testing.assert_allclose(default.means_, explicit.means_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(default.sigmas_, explicit.sigmas_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(default.weights_, explicit.weights_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        default.objective_, explicit.objective_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        default.optimization_results_[0]["history"],
        explicit.optimization_results_[0]["history"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(default.means_, observed.means_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(default.sigmas_, observed.sigmas_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(default.weights_, observed.weights_, rtol=0.0, atol=0.0)
    assert [snapshot.iteration for snapshot in snapshots] == [0, 1, 2]


def test_sequential_bandwidth_callbacks_form_one_ordered_timeline():
    data, init = _small_mixture()
    snapshots = []
    MomentGaussianMixtureModel(
        2,
        s_values=[1.4, 0.75],
        init=init,
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        amplitude_optimization="simplex",
        s_optimization="sequential",
        max_steps=1,
        objective_rtol=0.0,
        convergence_patience=99,
        random_state=29,
    ).fit(data, iteration_callback=snapshots.append)

    assert [snapshot.iteration for snapshot in snapshots] == [0, 1, 2, 3]
    assert [snapshot.metadata["bandwidth_index"] for snapshot in snapshots] == [
        0,
        0,
        1,
        1,
    ]
    assert [snapshot.phase for snapshot in snapshots] == [
        "initialization",
        "optimization",
        "initialization",
        "optimization",
    ]


@pytest.mark.parametrize(
    "override, match",
    [
        ({"s_optimization": "sequential"}, "s_optimization='joint'"),
        ({"amplitude_optimization": "non_negative"}, "amplitude_optimization"),
        ({"geometry_optimization": "joint"}, "geometry_optimization"),
        ({"joint_polish_sweeps": 1}, "joint polish"),
    ],
)
def test_separated_variance_modes_reject_unsupported_legacy_features(
    override,
    match,
):
    kwargs = {
        "n_components": 1,
        "s_values": [1.0],
        "init": {"means": [[0.0]], "sigmas": [1.0], "weights": [1.0]},
        "n_zero_moments": 0,
        "n_first_moments": 1,
        "n_second_moments": 0,
        "amplitude_optimization": "simplex",
        "s_optimization": "joint",
        "geometry_optimization": "component",
        "optimizer_mode": "joint_variance",
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        MomentGaussianMixtureModel(**kwargs)


def test_separated_variance_diagnostic_metrics_use_optimizer_result_counters():
    class Model:
        optimization_results_ = [
            {
                "initial_objective": 1.0,
                "history": np.array([0.8, 0.88, 0.7]),
                "variance_formula_updates": 10,
                "variance_formula_invalid_updates": 2,
                "variance_formula_q_clip_hits": 3,
                "variance_formula_eta_step_bound_hits": 4,
                "variance_joint_calls": 0,
            },
            {
                "initial_objective": 0.5,
                "history": np.array([0.4]),
                "variance_formula_updates": 0,
                "variance_formula_invalid_updates": 0,
                "variance_formula_q_clip_hits": 0,
                "variance_formula_eta_step_bound_hits": 0,
                "variance_joint_calls": 7,
            },
        ]

    model = Model()
    assert FormulaVarianceUpdates()(model) == 10
    assert FormulaVarianceInvalidUpdateRate()(model) == pytest.approx(0.2)
    assert FormulaVarianceQClipRate()(model) == pytest.approx(3 / 8)
    assert FormulaVarianceEtaStepBoundHitRate()(model) == pytest.approx(0.5)
    assert JointVarianceSweeps()(model) == 7
    assert OuterObjectiveIncreaseRate()(model) == pytest.approx(1 / 4)
    assert MaximumRelativeOuterObjectiveIncrease()(model) == pytest.approx(0.1)
