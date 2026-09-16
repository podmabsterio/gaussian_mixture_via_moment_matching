"""Canonicalization and validation of moment-estimator configuration."""

import numpy as np

from src_np.optimization.utils import RADIAL_SECOND_MOMENT


def canonical_second_moment_mode(mode):
    aliases = {
        "same": "same_direction",
        "same_direction": "same_direction",
        "squared": "same_direction",
        "isotropic": "same_direction",
        "independent": "two_directions",
        "two_directions": "two_directions",
        "cross": "two_directions",
        "anisotropic": "two_directions",
    }
    try:
        return aliases[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "second_moment_mode must be 'same_direction' or 'two_directions'"
        ) from exc


def canonical_s_optimization(mode):
    aliases = {
        "sequential": "sequential",
        "one_s": "sequential",
        "joint": "joint",
        "all": "joint",
        "multi_s": "joint",
    }
    try:
        return aliases[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError("s_optimization must be 'sequential' or 'joint'") from exc


def canonical_optimizer_mode(mode):
    aliases = {
        "legacy": "legacy",
        "joint_geometry": "legacy",
        "formula_variance": "formula_variance",
        "profiled_variance": "formula_variance",
        "joint_variance": "joint_variance",
        "alternating_joint_variance": "joint_variance",
    }
    try:
        return aliases[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "optimizer_mode must be 'legacy', 'formula_variance', or 'joint_variance'"
        ) from exc


def validate_configuration(model):
    if model.k < 1:
        raise ValueError("n_components must be positive")
    if model.n_init < 1:
        raise ValueError("n_init must be positive")
    counts = {
        0: model.n_zero_moments,
        1: model.n_first_moments,
        2: model.n_second_moments,
        RADIAL_SECOND_MOMENT: model.n_radial_second_moments,
    }
    if model.n_radial_second_moments not in (0, 1):
        raise ValueError("n_radial_second_moments must be zero or one")
    if any(count < 0 for count in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("moment counts must be non-negative and not all zero")
    if any(
        not np.isfinite(weight) or weight < 0
        for weight in model.moment_weights.values()
    ):
        raise ValueError("moment weights must be finite and non-negative")
    if not any(
        count > 0 and model.moment_weights[family] > 0
        for family, count in counts.items()
    ):
        raise ValueError("at least one enabled moment needs a positive weight")
    if model.geometry_optimization not in ("component", "joint"):
        raise ValueError("geometry_optimization must be 'component' or 'joint'")
    if model.amplitude_optimization not in ("non_negative", "simplex"):
        raise ValueError("amplitude_optimization must be 'non_negative' or 'simplex'")
    if model.test_center_mode not in ("data", "noisy_data"):
        raise ValueError("test_center_mode must be 'data' or 'noisy_data'")
    if model.n_test_centers is not None and model.n_test_centers < 1:
        raise ValueError("n_test_centers must be positive or None")
    if model.target_neighbor_fractions is not None:
        fractions = np.asarray(model.target_neighbor_fractions, dtype=float)
        if (
            fractions.size == 0
            or not np.all(np.isfinite(fractions))
            or np.any(fractions <= 0)
            or np.any(fractions >= 1)
        ):
            raise ValueError("target_neighbor_fractions must contain values in (0, 1)")
        if model.target_neighbor_counts is not None:
            raise ValueError(
                "target_neighbor_counts and target_neighbor_fractions are "
                "mutually exclusive"
            )
    if (
        not np.isfinite(model.test_center_fraction)
        or not 0 < model.test_center_fraction <= 1
    ):
        raise ValueError("test_center_fraction must lie in (0, 1]")
    if model.r_tests_near_means < 0 or model.alpha_std_tests < 0:
        raise ValueError("test-center sampling scales must be non-negative")
    if model.mean_penalty_weight < 0:
        raise ValueError("mean_penalty_weight must be non-negative")
    if (
        not np.isfinite(model.weight_entropy_regularization)
        or model.weight_entropy_regularization < 0
        or not np.isfinite(model.weight_gini_regularization)
        or model.weight_gini_regularization < 0
    ):
        raise ValueError("weight regularization strengths must be finite and non-negative")
    if (
        model.weight_entropy_regularization > 0
        or model.weight_gini_regularization > 0
    ) and model.amplitude_optimization != "simplex":
        raise ValueError("weight regularization requires amplitude_optimization='simplex'")
    if model.weight_regularization_schedule not in {
        "constant",
        "linear",
        "quadratic",
        "cosine",
    }:
        raise ValueError(
            "weight_regularization_schedule must be 'constant', 'linear', "
            "'quadratic', or 'cosine'"
        )
    if model.objective_rtol < 0 or model.objective_atol < 0:
        raise ValueError("objective tolerances must be non-negative")
    if model.convergence_patience < 1:
        raise ValueError("convergence_patience must be at least one")
    if model.max_steps < 0 or model.geom_sweeps < 1 or model.ls_max_nfev < 1:
        raise ValueError("optimization iteration counts are invalid")
    if model.active_tol < 0:
        raise ValueError("active_tol must be non-negative")
    if model.eta_step_bound is not None and model.eta_step_bound <= 0:
        raise ValueError("eta_step_bound must be positive or None")
    if model.m_step_bound is not None and model.m_step_bound <= 0:
        raise ValueError("m_step_bound must be positive or None")
    if model.joint_polish_sweeps < 0:
        raise ValueError("joint_polish_sweeps must be non-negative")
    if model.joint_polish_max_nfev is not None and model.joint_polish_max_nfev < 1:
        raise ValueError("joint_polish_max_nfev must be positive or None")
    if model.optimizer_mode != "legacy":
        if model.s_optimization != "joint":
            raise ValueError(
                "separated-variance optimizers require s_optimization='joint'"
            )
        if model.amplitude_optimization != "simplex":
            raise ValueError(
                "separated-variance optimizers require amplitude_optimization='simplex'"
            )
        if model.geometry_optimization != "component":
            raise ValueError(
                "separated-variance optimizers require "
                "geometry_optimization='component'"
            )
        if model.mean_penalty_weight != 0:
            raise ValueError(
                "separated-variance optimizers do not support a mean penalty"
            )
        if model.joint_polish_sweeps != 0 or model.joint_polish_max_nfev is not None:
            raise ValueError(
                "separated-variance optimizers do not support joint polish"
            )
    q_bounds = np.asarray(model.variance_formula_q_bounds, dtype=float)
    if (
        q_bounds.shape != (2,)
        or not np.all(np.isfinite(q_bounds))
        or not 0 < q_bounds[0] < q_bounds[1] < 1
    ):
        raise ValueError("variance_formula_q_bounds must satisfy 0 < low < high < 1")
    if (
        not np.isfinite(model.experimental_uniform_center_fraction)
        or not 0 < model.experimental_uniform_center_fraction < 1
    ):
        raise ValueError("experimental_uniform_center_fraction must lie in (0, 1)")
    if not (
        isinstance(model.test_center_noise_std, str)
        and model.test_center_noise_std == "auto"
    ):
        noise_std = np.asarray(model.test_center_noise_std, dtype=float)
        if (
            noise_std.ndim > 1
            or not np.all(np.isfinite(noise_std))
            or np.any(noise_std < 0)
        ):
            raise ValueError(
                "test_center_noise_std must be 'auto', a non-negative scalar, "
                "or a non-negative feature vector"
            )
