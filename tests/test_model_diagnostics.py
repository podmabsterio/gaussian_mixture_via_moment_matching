from types import SimpleNamespace

import numpy as np
import pytest

from experiments.metrics import (
    EstimatedVarianceRatio,
    FinalLogVarianceGradientInfinityNorm,
    FinalMeanGradientInfinityNorm,
    GeometryAcceptanceRate,
    GeometryEtaBoundHitRate,
    GeometryFunctionEvaluations,
    GeometryMaxEvaluationsRate,
    JointPolishRelativeGain,
    MaximumSelectedBandwidth,
    MinimumEstimatedWeight,
    MinimumSelectedBandwidth,
    OptimizationConverged,
    OptimizationOuterIterations,
    RelativeLossDrop,
    TrueRelativeObjectiveDrop,
)


def test_selected_bandwidth_diagnostics_summarize_moment_model_bandwidths():
    model = SimpleNamespace(s_values=[4.0, 1.5, 2.0])

    assert MinimumSelectedBandwidth()(model=model) == pytest.approx(1.5)
    assert MaximumSelectedBandwidth()(model=model) == pytest.approx(4.0)


def test_selected_bandwidth_diagnostics_are_nan_for_non_moment_model():
    model = SimpleNamespace()

    assert np.isnan(MinimumSelectedBandwidth()(model=model))
    assert np.isnan(MaximumSelectedBandwidth()(model=model))


def test_mixture_failure_diagnostics_use_fitted_parameters():
    weights = np.array([0.2, 0.3, 0.5])
    covariances = np.array([0.5 * np.eye(2), 2.0 * np.eye(2), np.eye(2)])

    assert MinimumEstimatedWeight()(weights=weights) == pytest.approx(0.2)
    assert EstimatedVarianceRatio()(covariances=covariances) == pytest.approx(4.0)


def test_optimizer_diagnostics_aggregate_internal_solver_results():
    model = SimpleNamespace(
        optimization_results_=[
            {
                "n_iter": 4,
                "n_outer_iter": 4,
                "converged": True,
                "geometry_calls": 12,
                "geometry_nfev": 48,
                "geometry_max_nfev_hits": 3,
                "geometry_accepted": 9,
                "geometry_eta_bound_hits": 2,
                "geometry_eta_coordinates": 12,
                "initial_objective": 2.0,
                "objective": 1.5,
                "joint_polish_relative_gain": 0.125,
                "final_mean_gradient_inf": 0.02,
                "final_eta_gradient_inf": 0.03,
            }
        ]
    )

    assert OptimizationOuterIterations()(model=model) == 4
    assert OptimizationConverged()(model=model) == 1
    assert GeometryFunctionEvaluations()(model=model) == 48
    assert GeometryMaxEvaluationsRate()(model=model) == pytest.approx(0.25)
    assert GeometryEtaBoundHitRate()(model=model) == pytest.approx(1 / 6)
    assert GeometryAcceptanceRate()(model=model) == pytest.approx(0.75)
    assert FinalMeanGradientInfinityNorm()(model=model) == pytest.approx(0.02)
    assert FinalLogVarianceGradientInfinityNorm()(model=model) == pytest.approx(0.03)
    assert JointPolishRelativeGain()(model=model) == pytest.approx(0.125)
    assert TrueRelativeObjectiveDrop()(model=model) == pytest.approx(0.25)
    assert RelativeLossDrop()(model=model) == pytest.approx(0.25)


def test_optimizer_diagnostics_are_nan_for_non_moment_baselines():
    model = SimpleNamespace()

    assert np.isnan(OptimizationOuterIterations()(model=model))
    assert np.isnan(OptimizationConverged()(model=model))
    assert np.isnan(GeometryFunctionEvaluations()(model=model))
    assert np.isnan(FinalMeanGradientInfinityNorm()(model=model))
