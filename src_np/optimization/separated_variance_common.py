"""Restricted optimizer core for experiments with separated variance updates.

This module intentionally does not replace or modify the legacy moment optimizer.
It implements only the configuration used by the recent moment experiments:

* all bandwidths are fitted jointly;
* mixture weights are optimized on the simplex;
* component means are updated one component at a time;
* there is no mean penalty or final joint polish.

The two public experimental algorithms live in ``formula_variance.py`` and
``joint_variance.py``.  Keeping their shared mechanics here avoids copying the
moment model, simplex solve, and diagnostics twice.
"""

import time

import numpy as np
from scipy.optimize import least_squares, minimize

from .moments import _prepare_moment_blocks
from .utils import MIN_SIGMA, _evaluate_moment_Q_and_jacobian


FORMULA_VARIANCE = "formula"
JOINT_VARIANCE = "joint"


def _validate_restricted_configuration(
    *,
    amplitude_optimization,
    geometry_optimization,
    mean_penalty_weight,
    joint_polish_sweeps,
    joint_polish_max_nfev,
):
    if amplitude_optimization != "simplex":
        raise ValueError(
            "separated-variance optimizers require amplitude_optimization='simplex'"
        )
    if geometry_optimization != "component":
        raise ValueError(
            "separated-variance optimizers require geometry_optimization='component'"
        )
    if float(mean_penalty_weight) != 0.0:
        raise ValueError("separated-variance optimizers do not support a mean penalty")
    if int(joint_polish_sweeps) != 0 or joint_polish_max_nfev is not None:
        raise ValueError("separated-variance optimizers do not support joint polish")


def _validate_formula_q_bounds(bounds):
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if (
        values.shape != (2,)
        or not np.all(np.isfinite(values))
        or not 0.0 < values[0] < values[1] < 1.0
    ):
        raise ValueError("variance_formula_q_bounds must satisfy 0 < low < high < 1")
    return float(values[0]), float(values[1])


def variance_from_corrected_radial_moments(
    corrected_zero,
    corrected_radial,
    s,
    *,
    q_bounds,
):
    """Apply equation (26) after neighboring contributions were subtracted.

    Returns ``(variance, raw_q, clipped_q)``.  The ratio restriction is the
    numerical stabilization proposed directly after equation (26).
    """
    corrected_zero = float(corrected_zero)
    corrected_radial = float(corrected_radial)
    s = float(s)
    q_low, q_high = _validate_formula_q_bounds(q_bounds)
    if not np.isfinite(corrected_zero) or not np.isfinite(corrected_radial):
        raise ValueError("corrected radial moments must be finite")
    if corrected_zero <= np.finfo(float).tiny:
        raise ValueError("corrected zeroth moment must be positive")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")

    raw_q = corrected_radial / corrected_zero
    if not np.isfinite(raw_q):
        raise ValueError("corrected radial ratio is not finite")
    clipped_q = float(np.clip(raw_q, q_low, q_high))
    variance = s * s * clipped_q / (1.0 - clipped_q)
    return float(variance), float(raw_q), clipped_q


def _fit_separated_variance_moment_gmm(
    moment_blocks,
    means_init,
    sigmas_init,
    *,
    variance_mode,
    data=None,
    amplitudes_init=None,
    weights_init=None,
    n_outer=50,
    geom_sweeps=1,
    ls_max_nfev=30,
    active_tol=1e-8,
    eta_step_bound=0.75,
    m_step_bound=None,
    joint_polish_sweeps=0,
    joint_polish_max_nfev=None,
    means_reference=None,
    mean_penalty_weight=0.0,
    objective_rtol=1e-6,
    objective_atol=0.0,
    convergence_patience=2,
    amplitude_optimization="simplex",
    geometry_optimization="component",
    variance_formula_q_bounds=(1e-4, 1.0 - 1e-4),
    verbose=False,
):
    """Fit a moment GMM while separating mean and variance updates."""
    if variance_mode not in (FORMULA_VARIANCE, JOINT_VARIANCE):
        raise ValueError("unknown separated variance mode")
    _validate_restricted_configuration(
        amplitude_optimization=amplitude_optimization,
        geometry_optimization=geometry_optimization,
        mean_penalty_weight=mean_penalty_weight,
        joint_polish_sweeps=joint_polish_sweeps,
        joint_polish_max_nfev=joint_polish_max_nfev,
    )

    means = np.asarray(means_init, dtype=float).copy()
    sigmas = np.asarray(sigmas_init, dtype=float).reshape(-1).copy()
    if means.ndim != 2 or means.shape[0] == 0:
        raise ValueError("means_init must be a non-empty two-dimensional array")
    K, d = means.shape
    if not np.all(np.isfinite(means)):
        raise ValueError("means_init must contain only finite values")
    if sigmas.shape != (K,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas_init must be finite and match means_init")
    sigmas = np.maximum(sigmas, MIN_SIGMA)

    if int(n_outer) < 0 or int(geom_sweeps) < 1 or int(ls_max_nfev) < 1:
        raise ValueError("optimization iteration counts are invalid")
    if active_tol < 0 or objective_rtol < 0 or objective_atol < 0:
        raise ValueError("optimization tolerances must be non-negative")
    if int(convergence_patience) < 1:
        raise ValueError("convergence_patience must be at least one")
    if eta_step_bound is not None and eta_step_bound <= 0:
        raise ValueError("eta_step_bound must be positive or None")
    if m_step_bound is not None and m_step_bound <= 0:
        raise ValueError("m_step_bound must be positive or None")

    q_bounds = _validate_formula_q_bounds(variance_formula_q_bounds)
    if variance_mode == FORMULA_VARIANCE:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != d:
            raise ValueError("formula variance updates require non-empty matching data")
        if not np.all(np.isfinite(data)):
            raise ValueError("formula variance update data must be finite")

    blocks, group_labels = _prepare_moment_blocks(moment_blocks, d)
    n_groups = len(group_labels)
    block_indices_per_group = [
        [i for i, block in enumerate(blocks) if block["group_index"] == group]
        for group in range(n_groups)
    ]
    group_bandwidths = np.asarray(
        [blocks[indices[0]]["s"] for indices in block_indices_per_group],
        dtype=float,
    )
    for group, indices in enumerate(block_indices_per_group):
        if not all(
            np.isclose(blocks[index]["s"], group_bandwidths[group]) for index in indices
        ):
            raise ValueError("simplex optimization requires one s per amplitude group")

    formula_s = float(np.min(group_bandwidths))
    min_variance = MIN_SIGMA * MIN_SIGMA

    def evaluate_block(block, means_, variances_, *, compute_jacobian):
        Q, jacobian = _evaluate_moment_Q_and_jacobian(
            block["test_centers"],
            block["test_directions"],
            block["second_test_directions"],
            means_,
            variances_,
            block["s"],
            block["moment_order"],
            compute_jacobian=compute_jacobian,
            moment_family=block["moment_family"],
            compensated=block["compensated"],
        )
        if block["moment_scale"] != 1.0:
            Q = block["moment_scale"] * Q
            if jacobian is not None:
                jacobian = block["moment_scale"] * jacobian
        return Q, jacobian

    def build_Q_list(means_, sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, min_variance)
        return [
            evaluate_block(
                block,
                means_,
                variances_,
                compute_jacobian=False,
            )[0]
            for block in blocks
        ]

    def amplitude_factors(sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, min_variance)
        return np.exp(
            -0.5 * d * np.log1p(variances_[None, :] / group_bandwidths[:, None] ** 2)
        )

    def amplitude_eta_derivatives(sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, min_variance)
        s2 = group_bandwidths[:, None] ** 2
        return -0.5 * d * variances_[None, :] / (s2 + variances_[None, :])

    def amplitudes_from_weights(weights_, sigmas_):
        return amplitude_factors(sigmas_) * weights_[None, :]

    explicit_amplitudes = amplitudes_init is not None
    if amplitudes_init is None:
        amplitudes_init_array = np.full((n_groups, K), 1.0 / K, dtype=float)
    else:
        amplitudes_init_array = np.asarray(amplitudes_init, dtype=float).copy()
        if amplitudes_init_array.ndim == 1:
            if amplitudes_init_array.shape != (K,):
                raise ValueError("amplitudes_init must match the component count")
            amplitudes_init_array = np.tile(
                amplitudes_init_array[None, :],
                (n_groups, 1),
            )
        if amplitudes_init_array.shape != (n_groups, K):
            raise ValueError(
                "amplitudes_init must have shape (n_amplitude_groups, n_components)"
            )
        if np.any(amplitudes_init_array < 0) or not np.all(
            np.isfinite(amplitudes_init_array)
        ):
            raise ValueError("amplitudes_init must be finite and non-negative")

    if weights_init is None:
        if explicit_amplitudes:
            raw_weights = amplitudes_init_array / amplitude_factors(sigmas)
            row_sums = np.sum(raw_weights, axis=1, keepdims=True)
            valid = row_sums[:, 0] > np.finfo(float).tiny
            if np.any(valid):
                rows = raw_weights[valid] / row_sums[valid]
                mixture_weights = np.mean(rows, axis=0)
                mixture_weights /= np.sum(mixture_weights)
            else:
                mixture_weights = np.full(K, 1.0 / K, dtype=float)
        else:
            mixture_weights = np.full(K, 1.0 / K, dtype=float)
    else:
        mixture_weights = np.asarray(weights_init, dtype=float).reshape(-1).copy()
        if mixture_weights.shape != (K,):
            raise ValueError("weights_init must match the component count")
        if np.any(mixture_weights < 0) or not np.all(np.isfinite(mixture_weights)):
            raise ValueError("weights_init must be finite and non-negative")
        total = np.sum(mixture_weights)
        if total <= np.finfo(float).tiny:
            raise ValueError("weights_init must have a positive sum")
        mixture_weights /= total

    def solve_weights(Q_list, sigmas_, weights_start):
        factors = amplitude_factors(sigmas_)
        design = np.vstack(
            [
                block["residual_scale"] * Q * factors[block["group_index"]][None, :]
                for block, Q in zip(blocks, Q_list)
            ]
        )
        target = np.concatenate(
            [block["residual_scale"] * block["Z"] for block in blocks]
        )
        if K == 1:
            weights = np.ones(1, dtype=float)
        else:
            start = np.maximum(np.asarray(weights_start, dtype=float), 0.0)
            start /= np.sum(start)

            def objective_(weights_):
                residual_ = design @ weights_ - target
                return 0.5 * np.dot(residual_, residual_)

            def jacobian_(weights_):
                return design.T @ (design @ weights_ - target)

            result = minimize(
                objective_,
                start,
                jac=jacobian_,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * K,
                constraints={
                    "type": "eq",
                    "fun": lambda weights_: np.sum(weights_) - 1.0,
                    "jac": lambda weights_: np.ones_like(weights_),
                },
                options={"ftol": 1e-12, "maxiter": max(200, 20 * K)},
            )
            if not result.success:
                raise RuntimeError(
                    "simplex-constrained amplitude optimization failed: "
                    f"{result.message}"
                )
            weights = np.clip(result.x, 0.0, 1.0)
            weights /= np.sum(weights)
        return amplitudes_from_weights(weights, sigmas_), weights

    def data_objective(Q_list, amplitudes_):
        total = 0.0
        for block, Q in zip(blocks, Q_list):
            residual = block["Z"] - Q @ amplitudes_[block["group_index"]]
            total += 0.5 * block["residual_scale"] ** 2 * np.dot(residual, residual)
        return float(total)

    def data_objective_per_order(Q_list, amplitudes_):
        values = {0: 0.0, 1: 0.0, 2: 0.0}
        for block, Q in zip(blocks, Q_list):
            residual = block["Z"] - Q @ amplitudes_[block["group_index"]]
            values[block["moment_order"]] += (
                0.5 * block["residual_scale"] ** 2 * float(np.dot(residual, residual))
            )
        return values

    def data_objective_per_family(Q_list, amplitudes_):
        values = {}
        for block, Q in zip(blocks, Q_list):
            residual = block["Z"] - Q @ amplitudes_[block["group_index"]]
            family = block["moment_family"]
            values.setdefault(family, 0.0)
            values[family] += (
                0.5 * block["residual_scale"] ** 2 * float(np.dot(residual, residual))
            )
        return values

    diagnostics = {
        "calls": 0,
        "nfev": 0,
        "njev": 0,
        "max_nfev_hits": 0,
        "accepted": 0,
        "eta_bound_hits": 0,
        "eta_coordinates": 0,
        "optimality_sum": 0.0,
        "optimality_max": 0.0,
    }

    def record_least_squares(result, accepted, *, eta_coordinates):
        diagnostics["calls"] += 1
        diagnostics["nfev"] += int(result.nfev)
        diagnostics["njev"] += int(result.njev or 0)
        diagnostics["max_nfev_hits"] += int(result.status == 0)
        diagnostics["accepted"] += int(accepted)
        diagnostics["eta_coordinates"] += int(eta_coordinates)
        if eta_coordinates:
            diagnostics["eta_bound_hits"] += int(
                np.count_nonzero(np.asarray(result.active_mask)[-eta_coordinates:])
            )
        optimality = float(result.optimality)
        diagnostics["optimality_sum"] += optimality
        diagnostics["optimality_max"] = max(
            diagnostics["optimality_max"],
            optimality,
        )

    formula_diagnostics = {
        "updates": 0,
        "invalid": 0,
        "q_clip_hits": 0,
        "eta_step_bound_hits": 0,
        "raw_q": [],
    }

    def component_is_active(amplitudes_, component):
        maximum = float(np.max(amplitudes_))
        return maximum > 0 and np.max(amplitudes_[:, component]) > active_tol * maximum

    def component_mean_update(component, Q_list, amplitudes_):
        nonlocal means
        if not component_is_active(amplitudes_, component):
            return Q_list
        targets = []
        for block, Q in zip(blocks, Q_list):
            group = block["group_index"]
            a = amplitudes_[group]
            targets.append(block["Z"] - Q @ a + a[component] * Q[:, component])

        mean_start = means[component].copy()
        lower = np.full(d, -np.inf)
        upper = np.full(d, np.inf)
        if m_step_bound is not None:
            lower = mean_start - m_step_bound
            upper = mean_start + m_step_bound
        variance = max(float(sigmas[component] ** 2), min_variance)
        cache = {"mean": None}

        def evaluate(mean):
            if cache["mean"] is not None and np.array_equal(mean, cache["mean"]):
                return cache["value"]
            q_values = []
            q_jacobians = []
            for block in blocks:
                q, jacobian = evaluate_block(
                    block,
                    mean[None, :],
                    np.array([variance]),
                    compute_jacobian=True,
                )
                q_values.append(q[:, 0])
                q_jacobians.append(jacobian[:, 0, :d])
            value = (q_values, q_jacobians)
            cache["mean"] = mean.copy()
            cache["value"] = value
            return value

        def residual(mean):
            q_values = evaluate(mean)[0]
            return np.concatenate(
                [
                    block["residual_scale"]
                    * (target - amplitudes_[block["group_index"], component] * q)
                    for block, target, q in zip(blocks, targets, q_values)
                ]
            )

        def residual_jacobian(mean):
            q_jacobians = evaluate(mean)[1]
            return np.vstack(
                [
                    -block["residual_scale"]
                    * amplitudes_[block["group_index"], component]
                    * jacobian
                    for block, jacobian in zip(blocks, q_jacobians)
                ]
            )

        result = least_squares(
            residual,
            mean_start,
            jac=residual_jacobian,
            bounds=(lower, upper),
            method="trf",
            max_nfev=int(ls_max_nfev),
        )
        q_new_values = evaluate(result.x)[0]
        old_local = 0.0
        new_local = 0.0
        for block, target, Q, q_new in zip(
            blocks,
            targets,
            Q_list,
            q_new_values,
        ):
            amplitude = amplitudes_[block["group_index"], component]
            scale2 = block["residual_scale"] ** 2
            old_residual = target - amplitude * Q[:, component]
            new_residual = target - amplitude * q_new
            old_local += 0.5 * scale2 * np.dot(old_residual, old_residual)
            new_local += 0.5 * scale2 * np.dot(new_residual, new_residual)
        accepted = new_local <= old_local
        record_least_squares(result, accepted, eta_coordinates=0)
        if accepted:
            means[component] = result.x
            for Q, q_new in zip(Q_list, q_new_values):
                Q[:, component] = q_new
        return Q_list

    def component_mean_sweep(Q_list, amplitudes_):
        for component in range(K):
            Q_list = component_mean_update(component, Q_list, amplitudes_)
        return Q_list

    def apply_formula_variance_update(component, Q_list, amplitudes_):
        nonlocal sigmas
        if not component_is_active(amplitudes_, component):
            return Q_list, amplitudes_

        s2 = formula_s * formula_s
        center = means[component]
        differences = data - center
        squared_distances = np.einsum("nd,nd->n", differences, differences)
        kernel = np.exp(-0.5 * squared_distances / s2)
        empirical_zero = float(np.mean(kernel))
        empirical_radial = float(np.mean(squared_distances * kernel / (d * s2)))

        variances = np.maximum(sigmas * sigmas, min_variance)
        offsets = means - center
        offset_norm2 = np.einsum("kd,kd->k", offsets, offsets)
        totals = s2 + variances
        full_zero = mixture_weights * np.exp(
            -0.5 * d * np.log1p(variances / s2) - 0.5 * offset_norm2 / totals
        )
        radial_factor = s2 * offset_norm2 / (d * totals * totals) + variances / totals
        full_radial = full_zero * radial_factor
        neighbor_mask = np.arange(K) != component
        corrected_zero = empirical_zero - np.sum(full_zero[neighbor_mask])
        corrected_radial = empirical_radial - np.sum(full_radial[neighbor_mask])

        formula_diagnostics["updates"] += 1
        try:
            variance_new, raw_q, clipped_q = variance_from_corrected_radial_moments(
                corrected_zero,
                corrected_radial,
                formula_s,
                q_bounds=q_bounds,
            )
        except ValueError:
            formula_diagnostics["invalid"] += 1
            return Q_list, amplitudes_

        formula_diagnostics["raw_q"].append(raw_q)
        formula_diagnostics["q_clip_hits"] += int(clipped_q != raw_q)
        old_eta = np.log(max(float(sigmas[component] ** 2), min_variance))
        candidate_eta = np.log(max(variance_new, min_variance))
        if eta_step_bound is not None:
            bounded_eta = float(
                np.clip(
                    candidate_eta,
                    old_eta - eta_step_bound,
                    old_eta + eta_step_bound,
                )
            )
            formula_diagnostics["eta_step_bound_hits"] += int(
                bounded_eta != candidate_eta
            )
            candidate_eta = bounded_eta
        sigmas[component] = np.sqrt(max(float(np.exp(candidate_eta)), min_variance))

        variance_array = np.array([sigmas[component] ** 2])
        for Q, block in zip(Q_list, blocks):
            q_new = evaluate_block(
                block,
                means[component : component + 1],
                variance_array,
                compute_jacobian=False,
            )[0]
            Q[:, component] = q_new[:, 0]
        amplitudes_[:, component] = (
            mixture_weights[component] * amplitude_factors(sigmas)[:, component]
        )
        return Q_list, amplitudes_

    def joint_variance_sweep(Q_list, amplitudes_):
        nonlocal sigmas
        eta_start = np.log(np.maximum(sigmas * sigmas, min_variance))
        lower = np.full(K, -np.inf)
        upper = np.full(K, np.inf)
        if eta_step_bound is not None:
            lower = eta_start - eta_step_bound
            upper = eta_start + eta_step_bound
        cache = {"eta": None}

        def evaluate(eta):
            if cache["eta"] is not None and np.array_equal(eta, cache["eta"]):
                return cache["value"]
            variances_ = np.maximum(np.exp(eta), min_variance)
            sigmas_ = np.sqrt(variances_)
            amplitudes_new = amplitudes_from_weights(mixture_weights, sigmas_)
            amplitude_derivatives = amplitudes_new * amplitude_eta_derivatives(sigmas_)
            Q_new_list = []
            jacobians = []
            for block in blocks:
                Q, jacobian = evaluate_block(
                    block,
                    means,
                    variances_,
                    compute_jacobian=True,
                )
                group = block["group_index"]
                prediction_jacobian = (
                    jacobian[:, :, -1] * amplitudes_new[group][None, :]
                    + Q * amplitude_derivatives[group][None, :]
                )
                Q_new_list.append(Q)
                jacobians.append(-block["residual_scale"] * prediction_jacobian)
            value = (sigmas_, amplitudes_new, Q_new_list, np.vstack(jacobians))
            cache["eta"] = eta.copy()
            cache["value"] = value
            return value

        def residual(eta):
            _, amplitudes_new, Q_new_list, _ = evaluate(eta)
            return np.concatenate(
                [
                    block["residual_scale"]
                    * (block["Z"] - Q @ amplitudes_new[block["group_index"]])
                    for block, Q in zip(blocks, Q_new_list)
                ]
            )

        result = least_squares(
            residual,
            eta_start,
            jac=lambda eta: evaluate(eta)[3],
            bounds=(lower, upper),
            method="trf",
            max_nfev=int(ls_max_nfev),
        )
        sigmas_new, amplitudes_new, Q_new_list, _ = evaluate(result.x)
        accepted = data_objective(Q_new_list, amplitudes_new) <= data_objective(
            Q_list,
            amplitudes_,
        )
        record_least_squares(result, accepted, eta_coordinates=K)
        if accepted:
            sigmas = sigmas_new.copy()
            return [Q.copy() for Q in Q_new_list], amplitudes_new.copy()
        return Q_list, amplitudes_

    def final_gradient_norms(amplitudes_):
        variances_ = np.maximum(sigmas * sigmas, min_variance)
        amplitude_derivatives = amplitudes_ * amplitude_eta_derivatives(sigmas)
        gradient = np.zeros((K, d + 1), dtype=float)
        for block in blocks:
            Q, jacobian = evaluate_block(
                block,
                means,
                variances_,
                compute_jacobian=True,
            )
            group = block["group_index"]
            prediction_jacobian = jacobian * amplitudes_[group][None, :, None]
            prediction_jacobian[:, :, -1] += Q * amplitude_derivatives[group][None, :]
            residual = block["residual_scale"] * (block["Z"] - Q @ amplitudes_[group])
            residual_jacobian = -block["residual_scale"] * prediction_jacobian
            gradient += np.einsum("jkp,j->kp", residual_jacobian, residual)
        return (
            float(np.max(np.abs(gradient[:, :d]))),
            float(np.max(np.abs(gradient[:, -1]))),
        )

    start_time = time.time()
    Q_list = build_Q_list(means, sigmas)
    amplitudes, mixture_weights = solve_weights(Q_list, sigmas, mixture_weights)
    previous_objective = data_objective(Q_list, amplitudes)
    initial_objective = previous_objective
    stable_iterations = 0
    history = []
    data_history = []

    iterator = range(int(n_outer))
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator)

    for _ in iterator:
        for _ in range(int(geom_sweeps)):
            if variance_mode == FORMULA_VARIANCE:
                # Formula updates are deliberately component-wise: optimize m_k
                # at fixed variances, then immediately profile sigma_k using (26).
                for component in range(K):
                    if not component_is_active(amplitudes, component):
                        continue
                    Q_list = component_mean_update(component, Q_list, amplitudes)
                    Q_list, amplitudes = apply_formula_variance_update(
                        component,
                        Q_list,
                        amplitudes,
                    )
            else:
                Q_list = component_mean_sweep(Q_list, amplitudes)
                Q_list, amplitudes = joint_variance_sweep(Q_list, amplitudes)

        amplitudes, mixture_weights = solve_weights(
            Q_list,
            sigmas,
            mixture_weights,
        )
        current_objective = data_objective(Q_list, amplitudes)
        history.append(current_objective)
        data_history.append(current_objective)

        improvement = previous_objective - current_objective
        tolerance = objective_atol + objective_rtol * max(
            abs(previous_objective),
            abs(current_objective),
            np.finfo(float).tiny,
        )
        if 0 <= improvement <= tolerance:
            stable_iterations += 1
        else:
            stable_iterations = 0
        if stable_iterations >= convergence_patience:
            break
        previous_objective = current_objective

    elapsed = time.time() - start_time
    final_objective = data_objective(Q_list, amplitudes)
    final_mean_gradient_inf, final_eta_gradient_inf = final_gradient_norms(amplitudes)
    n_outer_iter = len(history)
    calls = diagnostics["calls"]

    return {
        "means": means,
        "sigmas": sigmas,
        "amplitudes_per_group": amplitudes,
        "amplitudes": amplitudes[-1],
        "weights": mixture_weights,
        "amplitude_group_labels": group_labels,
        "objective": final_objective,
        "initial_objective": initial_objective,
        "history": np.asarray(history),
        "polish_history": np.asarray([]),
        "data_objective": final_objective,
        "data_history": np.asarray(data_history),
        "data_objective_per_order": data_objective_per_order(Q_list, amplitudes),
        "data_objective_per_family": data_objective_per_family(Q_list, amplitudes),
        "penalty_objective": 0.0,
        "penalty_history": np.zeros(n_outer_iter),
        "n_iter": n_outer_iter,
        "n_outer_iter": n_outer_iter,
        "converged": stable_iterations >= convergence_patience,
        "geometry_calls": calls,
        "geometry_nfev": diagnostics["nfev"],
        "geometry_njev": diagnostics["njev"],
        "geometry_max_nfev_hits": diagnostics["max_nfev_hits"],
        "geometry_accepted": diagnostics["accepted"],
        "geometry_eta_bound_hits": diagnostics["eta_bound_hits"],
        "geometry_eta_coordinates": diagnostics["eta_coordinates"],
        "geometry_max_nfev_fraction": (
            diagnostics["max_nfev_hits"] / calls if calls else 0.0
        ),
        "geometry_acceptance_rate": (diagnostics["accepted"] / calls if calls else 0.0),
        "geometry_eta_bound_fraction": (
            diagnostics["eta_bound_hits"] / diagnostics["eta_coordinates"]
            if diagnostics["eta_coordinates"]
            else 0.0
        ),
        "geometry_mean_optimality": (
            diagnostics["optimality_sum"] / calls if calls else 0.0
        ),
        "geometry_max_optimality": diagnostics["optimality_max"],
        "joint_polish_calls": 0,
        "joint_polish_nfev": 0,
        "pre_polish_objective": final_objective,
        "joint_polish_relative_gain": 0.0,
        "final_mean_gradient_inf": final_mean_gradient_inf,
        "final_eta_gradient_inf": final_eta_gradient_inf,
        "elapsed_seconds": elapsed,
        "loss_scales": np.asarray(
            [block["residual_scale"] for block in blocks],
            dtype=float,
        ),
        "variance_update_mode": variance_mode,
        "variance_formula_s": formula_s if variance_mode == FORMULA_VARIANCE else None,
        "variance_formula_updates": formula_diagnostics["updates"],
        "variance_formula_invalid_updates": formula_diagnostics["invalid"],
        "variance_formula_q_clip_hits": formula_diagnostics["q_clip_hits"],
        "variance_formula_eta_step_bound_hits": formula_diagnostics[
            "eta_step_bound_hits"
        ],
        "variance_formula_raw_q": np.asarray(formula_diagnostics["raw_q"]),
        "variance_joint_calls": (
            n_outer_iter * int(geom_sweeps) if variance_mode == JOINT_VARIANCE else 0
        ),
    }
