import numpy as np
import pytest

from experiments.metrics.metric_utils import sort_estimated_params_by_means
from experiments.metrics.weights_tv import WeightsTV
from src_np.gmm import MomentGaussianMixtureModel
from src_np.optimization.weights import (
    project_probability_simplex,
    solve_simplex_weights,
    weight_regularization_schedule_factor,
)


def test_probability_simplex_projection_is_feasible_and_idempotent():
    projected = project_probability_simplex(np.array([-2.0, 0.3, 1.7, 0.1]))

    assert np.all(projected >= 0)
    np.testing.assert_allclose(np.sum(projected), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        project_probability_simplex(projected),
        projected,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("constant", 1.0),
        ("linear", 0.75),
        ("quadratic", 0.9375),
        ("cosine", 0.5 * (1.0 + np.cos(0.25 * np.pi))),
    ],
)
def test_weight_regularization_schedule_factors(schedule, expected):
    assert weight_regularization_schedule_factor(schedule, 0.25) == pytest.approx(
        expected
    )
    assert weight_regularization_schedule_factor(schedule, 0.0) == 1.0
    expected_end = 1.0 if schedule == "constant" else 0.0
    assert weight_regularization_schedule_factor(schedule, 1.0) == pytest.approx(
        expected_end,
        abs=1e-15,
    )


def test_entropy_regularization_uses_momentum_pgd_and_can_sparse_weights():
    result = solve_simplex_weights(
        np.eye(4),
        np.full(4, 0.25),
        entropy_regularization=0.5,
        random_state=1,
    )

    assert result.method == "momentum_pgd"
    assert result.converged
    np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-14)
    assert np.count_nonzero(result.weights > 1e-10) == 1
    assert result.entropy_regularization_objective >= 0
    assert result.gini_regularization_objective == 0


def test_gini_regularization_keeps_quadratic_solver_and_can_sparse_weights():
    result = solve_simplex_weights(
        np.eye(4),
        np.full(4, 0.25),
        gini_regularization=0.6,
        random_state=1,
    )

    assert result.method == "quadratic_slsqp"
    np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-14)
    assert np.count_nonzero(result.weights > 1e-10) == 1
    assert result.entropy_regularization_objective == 0
    assert result.gini_regularization_objective < 0


def test_zero_regularization_preserves_unregularized_quadratic_solution():
    expected = np.array([0.15, 0.55, 0.30])
    result = solve_simplex_weights(np.eye(3), expected)

    assert result.method == "quadratic_slsqp"
    np.testing.assert_allclose(result.weights, expected, atol=1e-8)
    assert result.regularization_objective == 0


def test_weight_active_set_excludes_exactly_zero_components():
    design = np.eye(3)
    target = np.array([0.0, 1.0, 0.0])
    initial_weights = np.array([1.0, 0.0, 0.0])

    active_result = solve_simplex_weights(
        design,
        target,
        initial_weights,
        use_active_set=True,
    )
    unrestricted_result = solve_simplex_weights(
        design,
        target,
        initial_weights,
        use_active_set=False,
    )

    np.testing.assert_array_equal(active_result.weights, initial_weights)
    assert active_result.n_optimized_components == 1
    np.testing.assert_allclose(unrestricted_result.weights, target, atol=1e-8)
    assert unrestricted_result.n_optimized_components == 3


def test_weight_active_set_uses_relative_active_tolerance():
    design = np.eye(3)
    target = np.array([0.0, 1.0, 0.0])
    initial_weights = np.array([0.999, 0.001, 0.0])

    thresholded_result = solve_simplex_weights(
        design,
        target,
        initial_weights,
        use_active_set=True,
        active_tolerance=0.01,
    )
    zero_only_result = solve_simplex_weights(
        design,
        target,
        initial_weights,
        use_active_set=True,
        active_tolerance=0.0,
    )

    np.testing.assert_array_equal(thresholded_result.weights, [1.0, 0.0, 0.0])
    assert thresholded_result.n_optimized_components == 1
    np.testing.assert_allclose(zero_only_result.weights, target, atol=1e-8)
    assert zero_only_result.n_optimized_components == 2


def test_data_point_initialization_uses_all_points_and_positive_shared_scale():
    data = np.array([[-2.0, 0.0], [-1.0, 0.5], [1.0, -0.5], [2.0, 0.0]])
    model = MomentGaussianMixtureModel(
        # UI runners used to inject the oracle K here.  The data-points mode
        # must deliberately replace it with one component per observation.
        n_components=2,
        init="data_points",
        s_values=[1.0],
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        amplitude_optimization="simplex",
        max_steps=0,
        random_state=3,
    ).fit(data)

    np.testing.assert_array_equal(model.means_, data)
    assert model.k == model.n_components_ == len(data)
    np.testing.assert_array_equal(model.data_point_init_indices_, np.arange(len(data)))
    assert np.all(model.sigmas_ > 1e-6)
    np.testing.assert_allclose(model.sigmas_, model.sigmas_[0])
    np.testing.assert_allclose(np.sum(model.weights_), 1.0, atol=1e-14)


@pytest.mark.parametrize(
    "keyword",
    ["weight_entropy_regularization", "weight_gini_regularization"],
)
def test_weight_regularization_requires_simplex_weights(keyword):
    with pytest.raises(ValueError, match="requires amplitude_optimization='simplex'"):
        MomentGaussianMixtureModel(
            n_components=2,
            **{keyword: 0.1},
        )


def test_model_reports_regularized_objective_and_selected_solver():
    data = np.array([[-1.0], [-0.8], [0.8], [1.0]])
    model = MomentGaussianMixtureModel(
        n_components=4,
        init="data_points",
        s_values=[1.0],
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        amplitude_optimization="simplex",
        s_optimization="joint",
        weight_entropy_regularization=0.2,
        weight_gini_regularization=0.1,
        max_steps=0,
        random_state=5,
    ).fit(data)

    assert model.weight_solver_method_ == "momentum_pgd"
    assert model.weight_solver_active_components_ == 4
    np.testing.assert_array_equal(model.weight_solver_active_history_, [4])
    np.testing.assert_allclose(
        model.objective_,
        model.data_objective_ + model.penalty_objective_,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.weight_regularization_objective_,
        model.entropy_regularization_objective_
        + model.gini_regularization_objective_,
        atol=1e-12,
    )


def test_model_active_set_reduces_later_weight_problems_and_can_be_disabled():
    data = np.array([[-1.0], [-0.8], [0.8], [1.0]])
    common = {
        "n_components": 4,
        "init": "data_points",
        "s_values": [1.0],
        "n_zero_moments": 1,
        "n_first_moments": 0,
        "n_second_moments": 0,
        "amplitude_optimization": "simplex",
        "s_optimization": "joint",
        "weight_entropy_regularization": 0.2,
        "weight_gini_regularization": 0.1,
        "max_steps": 1,
        "random_state": 5,
    }

    active_model = MomentGaussianMixtureModel(
        **common,
        weight_active_set=True,
    ).fit(data)
    unrestricted_model = MomentGaussianMixtureModel(
        **common,
        weight_active_set=False,
    ).fit(data)

    np.testing.assert_array_equal(active_model.weight_solver_active_history_, [4, 1])
    np.testing.assert_array_equal(
        unrestricted_model.weight_solver_active_history_,
        [4, 4],
    )


def test_model_applies_regularization_schedule_over_outer_steps():
    data = np.array([[-1.0], [-0.8], [0.8], [1.0]])
    model = MomentGaussianMixtureModel(
        n_components=4,
        init="data_points",
        s_values=[1.0],
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        amplitude_optimization="simplex",
        s_optimization="joint",
        weight_entropy_regularization=0.01,
        weight_regularization_schedule="quadratic",
        weight_active_set=False,
        max_steps=4,
        convergence_patience=10,
        random_state=5,
    ).fit(data)

    np.testing.assert_allclose(
        model.weight_regularization_factor_history_,
        [1.0, 0.9375, 0.75, 0.4375, 0.0],
        atol=1e-15,
    )
    assert model.weight_regularization_factor_ == 0.0


def test_matching_retains_extra_components_for_rendering_and_density_metrics():
    true_means = np.array([[-1.0], [1.0]])
    means = np.array([[5.0], [1.1], [-0.9], [8.0]])
    weights = np.array([0.1, 0.35, 0.5, 0.05])
    covariances = np.arange(4, dtype=float)[:, None, None]

    ordered_means, ordered_weights, ordered_covariances = (
        sort_estimated_params_by_means(
            true_means,
            means,
            weights,
            covariances,
        )
    )

    np.testing.assert_array_equal(ordered_means[:, 0], [-0.9, 1.1, 5.0, 8.0])
    np.testing.assert_array_equal(ordered_weights, [0.5, 0.35, 0.1, 0.05])
    np.testing.assert_array_equal(ordered_covariances[:, 0, 0], [2.0, 1.0, 0.0, 3.0])
    assert WeightsTV()(ordered_weights, np.array([0.55, 0.45])) == pytest.approx(0.15)
