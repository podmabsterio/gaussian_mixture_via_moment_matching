from types import SimpleNamespace

import numpy as np
import pytest

from experiments.metrics import SmallBandwidthEmpiricalMomentByCenterRole
from src_np.gmm import MomentGaussianMixtureModel
from src_np.gmm.experimental_oracle_centers import (
    CENTER_ROLE_CLEAN,
    CENTER_ROLE_INDEPENDENT_UNIFORM,
    CENTER_ROLE_OBSERVED_OUTLIER,
    build_experimental_center_plan,
)


def _plan(X, mask, scenario, seed=4):
    return build_experimental_center_plan(
        X,
        scenario=scenario,
        contamination_mask=mask,
        background_lower=np.full(X.shape[1], -10.0),
        background_upper=np.full(X.shape[1], 10.0),
        uniform_fraction=0.05,
        rng=np.random.default_rng(seed),
    )


def test_oracle_center_plans_preserve_budget_and_roles():
    X = np.arange(40, dtype=float).reshape(20, 2)
    mask = np.zeros(20, dtype=bool)
    mask[[3, 11]] = True

    clean = _plan(X, mask, "oracle_clean_only")
    assert clean["centers"].shape == X.shape
    assert np.all(clean["roles"] == CENTER_ROLE_CLEAN)
    assert np.all(~mask[clean["source_indices"]])

    uniform = _plan(X, mask, "oracle_clean_plus_uniform")
    np.testing.assert_array_equal(
        np.flatnonzero(uniform["roles"] == CENTER_ROLE_INDEPENDENT_UNIFORM),
        np.flatnonzero(mask),
    )
    assert np.all(uniform["source_indices"][mask] == -1)
    assert np.all(uniform["centers"][mask] >= -10.0)
    assert np.all(uniform["centers"][mask] <= 10.0)

    observed = _plan(X, mask, "oracle_clean_plus_outliers")
    np.testing.assert_array_equal(observed["centers"], X)
    np.testing.assert_array_equal(observed["source_indices"], np.arange(20))
    np.testing.assert_array_equal(
        observed["roles"] == CENTER_ROLE_OBSERVED_OUTLIER,
        mask,
    )


def test_clean_uniform_plan_replaces_five_percent_without_growing_budget():
    X = np.arange(400, dtype=float).reshape(200, 2)
    mask = np.zeros(200, dtype=bool)
    plan = _plan(X, mask, "oracle_clean_plus_uniform")

    assert plan["centers"].shape == X.shape
    assert np.sum(plan["roles"] == CENTER_ROLE_INDEPENDENT_UNIFORM) == 10
    assert np.sum(plan["roles"] == CENTER_ROLE_CLEAN) == 190


def test_standard_and_oracle_observed_paths_are_bitwise_paired():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3))
    mask = np.zeros(20, dtype=bool)
    mask[[2, 13]] = True
    common = dict(
        n_components=1,
        s_values=[1.3],
        init={"means": [[0.0, 0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=1,
        n_first_moments=2,
        n_second_moments=2,
        leave_one_out=True,
        amplitude_optimization="simplex",
        experimental_oracle_test_centers=True,
        random_state=5,
        max_steps=0,
    )
    fit_kwargs = dict(
        X=X,
        contamination_mask=mask,
        background_lower=np.full(3, -10.0),
        background_upper=np.full(3, 10.0),
    )
    standard = MomentGaussianMixtureModel(**common).fit(
        **fit_kwargs,
        experimental_test_center_scenario="standard",
    )
    oracle = MomentGaussianMixtureModel(**common).fit(
        **fit_kwargs,
        experimental_test_center_scenario="oracle_clean_plus_outliers",
    )

    np.testing.assert_array_equal(
        standard.base_test_centers_, oracle.base_test_centers_
    )
    np.testing.assert_array_equal(standard.test_directions_, oracle.test_directions_)
    np.testing.assert_array_equal(
        standard.base_test_center_roles_, oracle.base_test_center_roles_
    )
    for key in standard.observed_moments_:
        np.testing.assert_array_equal(
            standard.observed_moments_[key], oracle.observed_moments_[key]
        )


def test_observed_bandwidth_reference_is_shared_with_clean_only_centers():
    rng = np.random.default_rng(19)
    X = np.vstack([rng.normal(size=(18, 2)), [[8.0, 8.0], [-9.0, 7.0]]])
    mask = np.zeros(20, dtype=bool)
    mask[-2:] = True
    common = dict(
        n_components=1,
        target_neighbor_counts=[3, 12],
        init={"means": [[0.0, 0.0]], "sigmas": [1.0], "weights": [1.0]},
        n_zero_moments=0,
        n_first_moments=0,
        n_second_moments=1,
        leave_one_out=True,
        amplitude_optimization="simplex",
        experimental_oracle_test_centers=True,
        random_state=3,
        max_steps=0,
    )
    fit_kwargs = dict(
        X=X,
        contamination_mask=mask,
        background_lower=np.full(2, -10.0),
        background_upper=np.full(2, 10.0),
        experimental_bandwidth_reference="all_observed",
    )
    standard = MomentGaussianMixtureModel(**common).fit(
        **fit_kwargs,
        experimental_test_center_scenario="standard",
    )
    clean_only = MomentGaussianMixtureModel(**common).fit(
        **fit_kwargs,
        experimental_test_center_scenario="oracle_clean_only",
    )
    clean_own_bandwidth = MomentGaussianMixtureModel(**common).fit(
        **{**fit_kwargs, "experimental_bandwidth_reference": "test_centers"},
        experimental_test_center_scenario="oracle_clean_only",
    )

    np.testing.assert_array_equal(standard.s_values, clean_only.s_values)
    assert not np.array_equal(
        standard.base_test_centers_, clean_only.base_test_centers_
    )
    assert not np.array_equal(standard.s_values, clean_own_bandwidth.s_values)


def test_small_bandwidth_moment_role_diagnostic_averages_directions():
    model = SimpleNamespace(
        s_values=[3.0, 1.0],
        base_test_center_roles_=np.array(
            [CENTER_ROLE_CLEAN, CENTER_ROLE_INDEPENDENT_UNIFORM]
        ),
        moment_test_functions_={2: {}},
        observed_moments_={
            (0, 2): np.array([100.0, 100.0, 100.0, 100.0]),
            (1, 2): np.array([1.0, 2.0, 3.0, 4.0]),
        },
    )
    metric = SmallBandwidthEmpiricalMomentByCenterRole(
        moment_order=2,
        center_role=CENTER_ROLE_INDEPENDENT_UNIFORM,
    )
    assert metric(model=model) == pytest.approx(3.0)


def test_small_bandwidth_moment_role_diagnostic_supports_radial_family():
    model = SimpleNamespace(
        s_values=[3.0, 1.0],
        base_test_center_roles_=np.array(
            [CENTER_ROLE_CLEAN, CENTER_ROLE_OBSERVED_OUTLIER]
        ),
        moment_test_functions_={"radial_second": {}},
        observed_moments_={
            (0, "radial_second"): np.array([100.0, 100.0]),
            (1, "radial_second"): np.array([-0.25, 0.75]),
        },
    )
    metric = SmallBandwidthEmpiricalMomentByCenterRole(
        moment_order=2,
        moment_family="radial_second",
        center_role=CENTER_ROLE_OBSERVED_OUTLIER,
    )
    assert metric(model=model) == pytest.approx(0.75)
