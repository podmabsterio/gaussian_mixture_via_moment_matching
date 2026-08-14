import numpy as np
import pytest

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.gmm import (
    MomentGaussianMixtureModel,
    MultiSGaussianMixtureModel,
    OneSGaussianMixtureModel,
)
from src_np.optimization import fit_dimension_free_moment_gmm
from src_np.optimization.utils import (
    RADIAL_SECOND_MOMENT,
    _evaluate_moment_Q_and_jacobian,
)
from src_np.test_functions_response import (
    compute_average_kernel_count,
    compute_kernel_counts,
    compute_Z_moment,
    compute_Z_radial_second_moment,
)


@pytest.mark.parametrize("moment_order", [0, 1, 2])
def test_moment_geometry_jacobian_matches_finite_differences(moment_order):
    rng = np.random.default_rng(12)
    n_tests, n_components, dimension = 5, 2, 3
    centers = rng.normal(size=(n_tests, dimension))
    directions = rng.normal(size=(n_tests, dimension))
    second_directions = rng.normal(size=(n_tests, dimension))
    means = rng.normal(size=(n_components, dimension))
    variances = np.exp(rng.normal(size=n_components))

    _, jacobian = _evaluate_moment_Q_and_jacobian(
        centers,
        directions,
        second_directions if moment_order == 2 else None,
        means,
        variances,
        1.4,
        moment_order,
        compute_jacobian=True,
    )

    theta = np.column_stack([means, np.log(variances)])
    numerical = np.empty_like(jacobian)
    epsilon = 1e-6
    for component in range(n_components):
        for parameter in range(dimension + 1):
            plus = theta.copy()
            minus = theta.copy()
            plus[component, parameter] += epsilon
            minus[component, parameter] -= epsilon
            q_plus = _evaluate_moment_Q_and_jacobian(
                centers,
                directions,
                second_directions if moment_order == 2 else None,
                plus[:, :dimension],
                np.exp(plus[:, -1]),
                1.4,
                moment_order,
                compute_jacobian=False,
            )[0]
            q_minus = _evaluate_moment_Q_and_jacobian(
                centers,
                directions,
                second_directions if moment_order == 2 else None,
                minus[:, :dimension],
                np.exp(minus[:, -1]),
                1.4,
                moment_order,
                compute_jacobian=False,
            )[0]
            numerical[:, component, parameter] = (
                q_plus[:, component] - q_minus[:, component]
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(jacobian, numerical, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("compensated", [False, True])
def test_radial_second_moment_jacobian_matches_finite_differences(compensated):
    rng = np.random.default_rng(112)
    n_tests, n_components, dimension = 6, 2, 4
    centers = rng.normal(size=(n_tests, dimension))
    means = rng.normal(size=(n_components, dimension))
    variances = np.exp(rng.normal(size=n_components))
    s = 1.2

    _, jacobian = _evaluate_moment_Q_and_jacobian(
        centers,
        None,
        None,
        means,
        variances,
        s,
        2,
        compute_jacobian=True,
        moment_family=RADIAL_SECOND_MOMENT,
        compensated=compensated,
    )

    theta = np.column_stack([means, np.log(variances)])
    numerical = np.empty_like(jacobian)
    epsilon = 1e-6
    for component in range(n_components):
        for parameter in range(dimension + 1):
            plus = theta.copy()
            minus = theta.copy()
            plus[component, parameter] += epsilon
            minus[component, parameter] -= epsilon
            q_plus = _evaluate_moment_Q_and_jacobian(
                centers,
                None,
                None,
                plus[:, :dimension],
                np.exp(plus[:, -1]),
                s,
                2,
                compute_jacobian=False,
                moment_family=RADIAL_SECOND_MOMENT,
                compensated=compensated,
            )[0]
            q_minus = _evaluate_moment_Q_and_jacobian(
                centers,
                None,
                None,
                minus[:, :dimension],
                np.exp(minus[:, -1]),
                s,
                2,
                compute_jacobian=False,
                moment_family=RADIAL_SECOND_MOMENT,
                compensated=compensated,
            )[0]
            numerical[:, component, parameter] = (
                q_plus[:, component] - q_minus[:, component]
            ) / (2.0 * epsilon)

    np.testing.assert_allclose(jacobian, numerical, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("moment_order", [0, 1, 2])
def test_empirical_moment_response_matches_direct_definition(moment_order):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(11, 4))
    centers = rng.normal(size=(7, 4))
    directions = rng.normal(size=(7, 4))
    second_directions = rng.normal(size=(7, 4))
    s = 0.9

    difference = data[:, None, :] - centers[None, :, :]
    kernel = np.exp(-np.sum(difference * difference, axis=2) / (2.0 * s * s))
    if moment_order == 0:
        expected = np.mean(kernel, axis=0)
    elif moment_order == 1:
        expected = np.mean(
            np.einsum("njd,jd->nj", difference, directions) * kernel,
            axis=0,
        )
    else:
        expected = np.mean(
            np.einsum("njd,jd->nj", difference, directions)
            * np.einsum("njd,jd->nj", difference, second_directions)
            * kernel,
            axis=0,
        )

    actual = compute_Z_moment(
        data,
        centers,
        directions,
        s,
        moment_order,
        second_test_directions=second_directions,
        data_batch_size=3,
        test_batch_size=2,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("moment_order", [0, 1, 2])
def test_empirical_moment_response_leave_one_out_matches_direct_definition(
    moment_order,
):
    rng = np.random.default_rng(41)
    data = rng.normal(size=(8, 3))
    leave_out_indices = np.array([5, 2, -1])
    centers = np.vstack(
        [
            data[5],
            data[2] + np.array([0.2, -0.1, 0.3]),
            rng.normal(size=3),
        ]
    )
    directions = rng.normal(size=centers.shape)
    second_directions = rng.normal(size=centers.shape)
    s = 1.1

    expected = []
    for center_index, (center, direction, second_direction) in enumerate(
        zip(centers, directions, second_directions)
    ):
        keep = np.ones(data.shape[0], dtype=bool)
        source_index = leave_out_indices[center_index]
        if source_index >= 0:
            keep[source_index] = False
        difference = data[keep] - center
        kernel = np.exp(-np.sum(difference * difference, axis=1) / (2.0 * s * s))
        if moment_order == 0:
            values = kernel
        elif moment_order == 1:
            values = (difference @ direction) * kernel
        else:
            values = (difference @ direction) * (difference @ second_direction) * kernel
        expected.append(np.mean(values))

    actual = compute_Z_moment(
        data,
        centers,
        directions,
        s,
        moment_order,
        second_test_directions=second_directions,
        data_batch_size=3,
        test_batch_size=2,
        leave_out_indices=leave_out_indices,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("compensated", [False, True])
def test_empirical_radial_second_moment_matches_direct_definition_and_loo(
    compensated,
):
    rng = np.random.default_rng(141)
    data = rng.normal(size=(9, 3))
    leave_out_indices = np.array([6, 2, -1, 4])
    centers = np.vstack(
        [
            data[6],
            data[2] + np.array([0.2, -0.1, 0.3]),
            rng.normal(size=3),
            data[4] + np.array([-0.1, 0.2, 0.1]),
        ]
    )
    s = 0.85

    expected = []
    for center_index, center in enumerate(centers):
        keep = np.ones(data.shape[0], dtype=bool)
        source_index = leave_out_indices[center_index]
        if source_index >= 0:
            keep[source_index] = False
        difference = data[keep] - center
        squared_radius = np.sum(difference * difference, axis=1)
        kernel = np.exp(-squared_radius / (2.0 * s * s))
        radial_factor = squared_radius / (data.shape[1] * s * s)
        if compensated:
            radial_factor -= 1.0
        expected.append(np.mean(radial_factor * kernel))

    actual = compute_Z_radial_second_moment(
        data,
        centers,
        s,
        compensated=compensated,
        data_batch_size=4,
        test_batch_size=2,
        leave_out_indices=leave_out_indices,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("compensated", [False, True])
def test_model_fits_with_only_radial_second_moments(compensated):
    rng = np.random.default_rng(151)
    data = rng.normal(loc=[0.4, -0.2], scale=0.7, size=(24, 2))
    s = 1.1
    model = MomentGaussianMixtureModel(
        1,
        s_values=[s],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=0,
        n_first_moments=0,
        n_second_moments=0,
        n_radial_second_moments=1,
        compensated_radial_second_moments=compensated,
        scale_normalize_moments=False,
        max_steps=2,
        random_state=17,
    ).fit(data)

    block = model.moment_test_functions_[RADIAL_SECOND_MOMENT]
    expected = compute_Z_radial_second_moment(
        data,
        block["test_centers"],
        s,
        compensated=compensated,
    )
    np.testing.assert_allclose(
        model.observed_moments_[0, RADIAL_SECOND_MOMENT],
        expected,
        rtol=2e-13,
        atol=2e-13,
    )
    assert np.isfinite(model.objective_)
    assert np.all(np.isfinite(model.means_))
    assert np.all(np.isfinite(model.sigmas_))
    assert RADIAL_SECOND_MOMENT in model.data_objective_per_family_
    assert model.data_objective_per_order_[2] == pytest.approx(
        model.data_objective_per_family_[RADIAL_SECOND_MOMENT]
    )


def test_leave_one_out_bandwidth_targets_exclude_anchor_observations():
    data = np.linspace(-3.0, 3.0, 15)[:, None]
    leave_out_indices = np.arange(data.shape[0])
    target = 4.0

    s = select_s_values_by_average_kernel_count(
        data,
        n_components=3,
        target_neighbor_counts=[target],
        test_centers=data,
        leave_out_indices=leave_out_indices,
    )[0]
    actual_count = compute_average_kernel_count(
        data,
        data,
        s,
        leave_out_indices=leave_out_indices,
    )

    assert actual_count == pytest.approx(target, rel=2e-3)


def test_moment_model_applies_leave_one_out_to_every_order():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(9, 2))
    s = 1.3
    model = MomentGaussianMixtureModel(
        1,
        s_values=[s],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=1,
        n_second_moments=1,
        leave_one_out=True,
        max_steps=0,
        random_state=3,
    ).fit(data)

    for order, block in model.moment_test_functions_.items():
        np.testing.assert_array_equal(
            block["leave_out_indices"],
            np.arange(data.shape[0]),
        )
        expected = compute_Z_moment(
            data,
            block["test_centers"],
            block["test_directions"],
            s,
            order,
            second_test_directions=block["second_test_directions"],
            leave_out_indices=block["leave_out_indices"],
        ) / (s**order)
        np.testing.assert_allclose(
            model.observed_moments_[0, order],
            expected,
            rtol=2e-13,
            atol=2e-13,
        )


def test_moment_model_applies_leave_one_out_to_automatic_bandwidth_selection():
    data = np.linspace(-3.0, 3.0, 15)[:, None]
    target = 4.0
    model = MomentGaussianMixtureModel(
        1,
        target_neighbor_counts=[target],
        init={"means": [[0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        leave_one_out=True,
        max_steps=0,
    ).fit(data)

    actual_count = compute_average_kernel_count(
        data,
        model.base_test_centers_,
        model.s_values[0],
        leave_out_indices=model.base_leave_out_indices_,
    )

    np.testing.assert_array_equal(
        model.base_leave_out_indices_,
        np.arange(data.shape[0]),
    )
    assert actual_count == pytest.approx(target, rel=2e-3)


def test_moment_model_scales_relative_bandwidth_targets_with_sample_size():
    data = np.linspace(-4.0, 4.0, 31)[:, None]
    fractions = [0.1, 0.6]
    model = MomentGaussianMixtureModel(
        1,
        target_neighbor_fractions=fractions,
        init={"means": [[0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        n_test_centers=11,
        leave_one_out=True,
        max_steps=0,
        random_state=4,
    ).fit(data)

    expected_targets = (data.shape[0] - 1) * np.asarray(fractions)
    actual_counts = [
        compute_average_kernel_count(
            data,
            model.base_test_centers_,
            s,
            leave_out_indices=model.base_leave_out_indices_,
        )
        for s in model.s_values
    ]

    np.testing.assert_allclose(actual_counts, expected_targets[::-1], rtol=2e-3)


def test_moment_model_rejects_conflicting_absolute_and_relative_bandwidth_targets():
    with pytest.raises(ValueError, match="mutually exclusive"):
        MomentGaussianMixtureModel(
            1,
            target_neighbor_counts=[3],
            target_neighbor_fractions=[0.25],
        )


@pytest.mark.parametrize("moment_order", [0, 1, 2])
def test_moment_model_scale_normalizes_responses_by_default(moment_order):
    rng = np.random.default_rng(19)
    data = rng.normal(size=(9, 3))
    s = 2.5
    common = {
        "n_components": 1,
        "s_values": [s],
        "init": {"means": [[0.0, 0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        "n_zero_moments": int(moment_order == 0),
        "n_first_moments": int(moment_order == 1),
        "n_second_moments": int(moment_order == 2),
        "max_steps": 0,
        "random_state": 7,
    }
    normalized = MomentGaussianMixtureModel(**common).fit(data)
    unscaled = MomentGaussianMixtureModel(
        **common,
        scale_normalize_moments=False,
    ).fit(data)

    block = normalized.moment_test_functions_[moment_order]
    raw_response = compute_Z_moment(
        data,
        block["test_centers"],
        block["test_directions"],
        s,
        moment_order,
        second_test_directions=block["second_test_directions"],
    )

    assert normalized.scale_normalize_moments is True
    assert unscaled.scale_normalize_moments is False
    np.testing.assert_allclose(
        normalized.observed_moments_[0, moment_order],
        raw_response / (s**moment_order),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        unscaled.observed_moments_[0, moment_order],
        raw_response,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("moment_order", [0, 1, 2])
def test_scale_normalized_moments_are_invariant_to_coordinate_rescaling(
    moment_order,
):
    rng = np.random.default_rng(29)
    data = rng.normal(size=(13, 3))
    centers = rng.normal(size=(7, 3))
    directions = rng.normal(size=(7, 3))
    second_directions = rng.normal(size=(7, 3))
    means = rng.normal(size=(2, 3))
    variances = np.exp(rng.normal(size=2))
    s = 1.3
    coordinate_scale = 4.2

    empirical = compute_Z_moment(
        data,
        centers,
        directions,
        s,
        moment_order,
        second_test_directions=second_directions,
    ) / (s**moment_order)
    rescaled_empirical = compute_Z_moment(
        coordinate_scale * data,
        coordinate_scale * centers,
        directions,
        coordinate_scale * s,
        moment_order,
        second_test_directions=second_directions,
    ) / ((coordinate_scale * s) ** moment_order)
    np.testing.assert_allclose(
        rescaled_empirical,
        empirical,
        rtol=2e-13,
        atol=2e-13,
    )

    analytical = _evaluate_moment_Q_and_jacobian(
        centers,
        directions,
        second_directions if moment_order == 2 else None,
        means,
        variances,
        s,
        moment_order,
        compute_jacobian=False,
    )[0] / (s**moment_order)
    rescaled_analytical = _evaluate_moment_Q_and_jacobian(
        coordinate_scale * centers,
        directions,
        second_directions if moment_order == 2 else None,
        coordinate_scale * means,
        coordinate_scale**2 * variances,
        coordinate_scale * s,
        moment_order,
        compute_jacobian=False,
    )[0] / ((coordinate_scale * s) ** moment_order)
    np.testing.assert_allclose(
        rescaled_analytical,
        analytical,
        rtol=2e-13,
        atol=2e-13,
    )


def test_kernel_counts_match_direct_definition():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(13, 4))
    centers = rng.normal(size=(7, 4))
    s = 0.8

    differences = data[:, None, :] - centers[None, :, :]
    expected = np.sum(
        np.exp(-np.sum(differences * differences, axis=2) / (2.0 * s * s)),
        axis=0,
    )
    actual = compute_kernel_counts(
        data,
        centers,
        s,
        data_batch_size=4,
        test_batch_size=3,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_test_center_fraction_validation(fraction):
    with pytest.raises(ValueError, match="test_center_fraction"):
        MomentGaussianMixtureModel(1, test_center_fraction=fraction)


def test_dense_center_filter_excludes_isolated_outliers():
    dense = np.array(
        [
            [-0.10, 0.00],
            [-0.05, 0.04],
            [0.00, -0.03],
            [0.03, 0.02],
            [0.06, -0.01],
            [0.09, 0.03],
            [-0.02, -0.07],
            [0.04, 0.08],
        ]
    )
    outliers = np.array([[20.0, 20.0], [-25.0, 18.0]])
    data = np.vstack([dense, outliers])

    model = MomentGaussianMixtureModel(
        1,
        s_values=[0.5, 1.0],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        test_center_fraction=0.8,
        max_steps=0,
        random_state=2,
    ).fit(data)

    np.testing.assert_array_equal(
        model.selected_data_center_indices_,
        np.arange(dense.shape[0]),
    )
    np.testing.assert_array_equal(model.data_test_centers_, dense)
    assert model.test_center_selection_s_ == 0.5
    assert np.min(model.data_center_kernel_counts_[: dense.shape[0]]) > np.max(
        model.data_center_kernel_counts_[dense.shape[0] :]
    )
    assert model.base_test_centers_.shape == dense.shape


def test_center_fraction_uses_ceiling_and_works_with_automatic_bandwidths():
    rng = np.random.default_rng(6)
    data = np.vstack(
        [
            rng.normal(0.0, 0.1, size=(9, 2)),
            np.array([[15.0, -12.0], [-18.0, 11.0]]),
        ]
    )
    model = MomentGaussianMixtureModel(
        1,
        target_neighbor_counts=[3.0],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        test_center_fraction=0.5,
        max_steps=0,
        random_state=4,
    ).fit(data)

    assert model.data_test_centers_.shape == (6, 2)
    assert model.selected_data_center_indices_.shape == (6,)
    assert model.selected_data_center_fraction_ == 6 / 11
    assert np.all(model.selected_data_center_indices_ < 9)
    assert model.test_center_selection_s_ == model.s_values[0]
    assert model.data_center_kernel_counts_.shape == (data.shape[0],)


def test_center_fraction_filters_before_sampling_and_keeps_near_mean_centers():
    rng = np.random.default_rng(7)
    data = np.vstack(
        [
            rng.normal(0.0, 0.1, size=(10, 2)),
            np.array([[12.0, 12.0], [-14.0, 13.0]]),
        ]
    )
    model = MomentGaussianMixtureModel(
        1,
        s_values=[0.5],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=0,
        n_second_moments=0,
        test_center_fraction=0.75,
        n_test_centers=4,
        r_tests_near_means=0.25,
        max_steps=0,
        random_state=4,
    ).fit(data)

    # Nine dense candidates survive filtering, four are sampled as data
    # centers, and ceil(0.25 * 12) = 3 near-mean centers are added afterwards.
    assert model.selected_data_center_indices_.shape == (9,)
    assert model.data_test_centers_.shape == (4, 2)
    assert model.base_test_centers_.shape == (7, 2)
    assert np.all(np.isin(model.sampled_data_center_indices_, np.arange(10)))


def _comparison_data_and_init():
    rng = np.random.default_rng(14)
    data = np.vstack(
        [
            rng.normal(-1.0, 0.6, size=(24, 2)),
            rng.normal(1.0, 0.8, size=(24, 2)),
        ]
    )
    init = {
        "means": np.array([[-0.8, -0.9], [0.8, 0.9]]),
        "sigmas": np.array([0.9, 0.9]),
        "amplitudes": np.array([0.2, 0.2]),
    }
    return data, init


@pytest.mark.parametrize("geometry_optimization", ["component", "joint"])
def test_first_order_sequential_configuration_reproduces_one_s_model(
    geometry_optimization,
):
    data, init = _comparison_data_and_init()
    old = OneSGaussianMixtureModel(
        2,
        s_values=[1.5, 0.9],
        init=init,
        num_directions=2,
        random_state=8,
        max_steps=2,
        objective_rtol=0.0,
        convergence_patience=99,
        joint_optimization=geometry_optimization == "joint",
    )
    new = MomentGaussianMixtureModel(
        2,
        s_values=[1.5, 0.9],
        init=init,
        n_zero_moments=0,
        n_first_moments=2,
        n_second_moments=0,
        normalize_moment_losses=False,
        scale_normalize_moments=False,
        s_optimization="sequential",
        geometry_optimization=geometry_optimization,
        random_state=8,
        max_steps=2,
        objective_rtol=0.0,
        convergence_patience=99,
    )

    old.fit(data)
    assert new.fit(data) is new
    np.testing.assert_allclose(new.test_directions_, old.test_directions_)
    np.testing.assert_allclose(new.means_, old.means_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.sigmas_, old.sigmas_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        new.amplitudes_,
        old.amplitudes_,
        rtol=1e-12,
        atol=1e-12,
    )


def test_first_order_joint_configuration_reproduces_multi_s_model():
    data, init = _comparison_data_and_init()
    old = MultiSGaussianMixtureModel(
        2,
        s_values=[1.5, 0.9],
        init=init,
        num_directions=2,
        random_state=8,
        max_steps=2,
        objective_rtol=0.0,
        convergence_patience=99,
    )
    new = MomentGaussianMixtureModel(
        2,
        s_values=[1.5, 0.9],
        init=init,
        n_zero_moments=0,
        n_first_moments=2,
        n_second_moments=0,
        normalize_moment_losses=False,
        scale_normalize_moments=False,
        s_optimization="joint",
        geometry_optimization="joint",
        random_state=8,
        max_steps=2,
        objective_rtol=0.0,
        convergence_patience=99,
    )

    old.fit(data)
    new.fit(data)
    np.testing.assert_allclose(new.means_, old.means_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new.sigmas_, old.sigmas_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        new.amplitudes_per_s_,
        old.amplitudes_per_s_,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(new.weights_, old.weights_, rtol=1e-12, atol=1e-12)


def test_each_moment_loss_uses_its_weight_and_own_test_count():
    rng = np.random.default_rng(21)
    data = rng.normal(size=(8, 2))
    model = MomentGaussianMixtureModel(
        1,
        s_values=[1.0],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=2,
        n_second_moments=3,
        zero_moment_weight=2.0,
        first_moment_weight=3.0,
        second_moment_weight=5.0,
        normalize_moment_losses=True,
        max_steps=0,
        random_state=1,
    ).fit(data)

    expected_scales = np.sqrt(
        np.array(
            [
                2.0 / (1 * data.shape[0]),
                3.0 / (2 * data.shape[0]),
                5.0 / (3 * data.shape[0]),
            ]
        )
    )
    np.testing.assert_allclose(
        model.optimization_results_[0]["loss_scales"],
        expected_scales,
    )


def test_simplex_optimizer_recovers_one_weight_vector_for_all_bandwidths():
    rng = np.random.default_rng(31)
    dimension = 3
    means = np.array([[-1.0, 0.2, 0.0], [0.4, 0.8, -0.3], [1.2, -0.5, 0.7]])
    sigmas = np.array([0.5, 0.8, 1.1])
    expected_weights = np.array([0.2, 0.5, 0.3])
    blocks = []

    for group, s in enumerate([1.7, 0.9]):
        factors = np.exp(-0.5 * dimension * np.log1p((sigmas * sigmas) / (s * s)))
        amplitudes = expected_weights * factors
        for moment_order in (0, 1, 2):
            centers = rng.normal(size=(18, dimension))
            directions = rng.normal(size=(18, dimension))
            second_directions = rng.normal(size=(18, dimension))
            Q = _evaluate_moment_Q_and_jacobian(
                centers,
                directions,
                second_directions if moment_order == 2 else None,
                means,
                sigmas * sigmas,
                s,
                moment_order,
                compute_jacobian=False,
            )[0]
            blocks.append(
                {
                    "moment_order": moment_order,
                    "test_centers": centers,
                    "test_directions": directions,
                    "second_test_directions": second_directions,
                    "Z": Q @ amplitudes,
                    "s": s,
                    "moment_scale": s ** (-moment_order),
                    "amplitude_group": group,
                }
            )

    result = fit_dimension_free_moment_gmm(
        blocks,
        means_init=means,
        sigmas_init=sigmas,
        amplitude_optimization="simplex",
        n_outer=0,
    )

    np.testing.assert_allclose(result["weights"], expected_weights, atol=2e-7)
    np.testing.assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-14)
    assert np.all(result["weights"] >= 0)
    for group, s in enumerate([1.7, 0.9]):
        expected_amplitudes = expected_weights * np.exp(
            -0.5 * dimension * np.log1p((sigmas * sigmas) / (s * s))
        )
        np.testing.assert_allclose(
            result["amplitudes_per_group"][group],
            expected_amplitudes,
            atol=2e-7,
        )


def test_joint_polish_starts_from_component_solution_and_records_diagnostics():
    rng = np.random.default_rng(52)
    data = np.vstack(
        [
            rng.normal(-0.8, 0.7, size=(24, 3)),
            rng.normal(0.8, 0.9, size=(24, 3)),
        ]
    )
    common = {
        "n_components": 2,
        "s_values": [1.3, 0.8],
        "n_init": 2,
        "n_zero_moments": 0,
        "n_first_moments": 0,
        "n_second_moments": 2,
        "amplitude_optimization": "simplex",
        "s_optimization": "joint",
        "geometry_optimization": "component",
        "max_steps": 3,
        "ls_max_nfev": 6,
        "objective_rtol": 0.0,
        "convergence_patience": 99,
        "random_state": 7,
    }

    baseline = MomentGaussianMixtureModel(**common).fit(data)
    polished = MomentGaussianMixtureModel(
        **common,
        joint_polish_sweeps=2,
        joint_polish_max_nfev=8,
    ).fit(data)

    baseline_result = baseline.optimization_results_[0]
    polished_result = polished.optimization_results_[0]
    np.testing.assert_allclose(
        polished_result["pre_polish_objective"],
        baseline_result["objective"],
        rtol=1e-11,
        atol=1e-14,
    )
    assert polished.objective_ <= baseline.objective_ + 1e-14
    assert polished_result["joint_polish_calls"] == 2
    assert polished_result["joint_polish_nfev"] > 0
    assert polished_result["geometry_calls"] == 3 * 2 + 2
    assert polished_result["geometry_eta_coordinates"] == 3 * 2 + 2 * 2
    assert np.isfinite(polished_result["final_mean_gradient_inf"])
    assert np.isfinite(polished_result["final_eta_gradient_inf"])
    np.testing.assert_allclose(
        polished_result["history"],
        baseline_result["history"],
        rtol=1e-11,
        atol=1e-14,
    )


@pytest.mark.parametrize("s_optimization", ["sequential", "joint"])
@pytest.mark.parametrize("geometry_optimization", ["component", "joint"])
def test_simplex_model_keeps_weights_and_amplitudes_feasible(
    s_optimization,
    geometry_optimization,
):
    rng = np.random.default_rng(32)
    data = np.vstack(
        [
            rng.normal(-1.0, 0.5, size=(15, 2)),
            rng.normal(1.0, 0.7, size=(10, 2)),
        ]
    )
    model = MomentGaussianMixtureModel(
        2,
        s_values=[1.4, 0.8],
        n_init=2,
        amplitude_optimization="simplex",
        s_optimization=s_optimization,
        geometry_optimization=geometry_optimization,
        max_steps=2,
        ls_max_nfev=5,
        random_state=4,
    ).fit(data)

    np.testing.assert_allclose(np.sum(model.weights_), 1.0, atol=1e-14)
    assert np.all(model.weights_ >= 0)
    last_s = model.s_values[-1]
    last_factors = np.exp(
        -0.5
        * data.shape[1]
        * np.log1p((model.sigmas_ * model.sigmas_) / (last_s * last_s))
    )
    np.testing.assert_allclose(
        model.amplitudes_,
        model.weights_ * last_factors,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        model.weights_per_s_,
        np.tile(model.weights_[None, :], (len(model.s_values), 1)),
    )
    for group, s in enumerate(model.s_values):
        factors = np.exp(
            -0.5 * data.shape[1] * np.log1p((model.sigmas_ * model.sigmas_) / (s * s))
        )
        np.testing.assert_allclose(
            model.amplitudes_per_s_[group],
            model.weights_ * factors,
            rtol=1e-12,
            atol=1e-12,
        )

    reconstructed_data_objective = 0.0
    if s_optimization == "joint":
        fitted_s_indices = range(len(model.s_values))
    else:
        fitted_s_indices = [len(model.s_values) - 1]
    for s_index in fitted_s_indices:
        s = model.s_values[s_index]
        for order, block in model.moment_test_functions_.items():
            Q = _evaluate_moment_Q_and_jacobian(
                block["test_centers"],
                block["test_directions"],
                block["second_test_directions"],
                model.means_,
                model.sigmas_ * model.sigmas_,
                s,
                order,
                compute_jacobian=False,
            )[0]
            if model.scale_normalize_moments:
                Q = Q / (s**order)
            residual = (
                model.observed_moments_[s_index, order]
                - Q @ model.amplitudes_per_s_[s_index]
            )
            denominator = residual.size if model.normalize_moment_losses else 1
            reconstructed_data_objective += (
                0.5
                * model.moment_weights[order]
                / denominator
                * np.dot(residual, residual)
            )
    np.testing.assert_allclose(
        model.data_objective_,
        reconstructed_data_objective,
        rtol=1e-11,
        atol=1e-13,
    )


@pytest.mark.parametrize("second_moment_mode", ["same_direction", "two_directions"])
def test_second_moment_modes_and_automatic_noisy_centers(second_moment_mode):
    rng = np.random.default_rng(5)
    data = np.vstack(
        [
            rng.normal(-1.0, 0.4, size=(15, 3)),
            rng.normal(1.0, 0.5, size=(15, 3)),
        ]
    )
    model = MomentGaussianMixtureModel(
        2,
        s_values=[1.2],
        n_init=2,
        n_zero_moments=1,
        n_first_moments=1,
        n_second_moments=1,
        second_moment_mode=second_moment_mode,
        test_center_mode="noisy_data",
        test_center_noise_std="auto",
        n_test_centers=12,
        random_state=3,
        max_steps=1,
        ls_max_nfev=4,
    ).fit(data)

    assert model.test_center_noise_std_ > 0
    assert model.moment_test_functions_[2]["test_centers"].shape == (12, 3)
    first = model.moment_test_functions_[2]["test_directions"]
    second = model.moment_test_functions_[2]["second_test_directions"]
    if second_moment_mode == "same_direction":
        np.testing.assert_array_equal(first, second)
    else:
        assert not np.array_equal(first, second)
    np.testing.assert_allclose(np.sum(model.weights_), 1.0)
    assert all(order in model.data_objective_per_order_ for order in (0, 1, 2))
