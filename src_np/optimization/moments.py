import time
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares, minimize, nnls

from src_np.iteration import IterationSnapshot, MixtureParameters

from .utils import (
    MIN_SIGMA,
    RADIAL_SECOND_MOMENT,
    _build_mean_penalty_jacobian,
    _evaluate_moment_Q_and_jacobian,
    _pack_geometry,
    _unpack_geometry,
)


def _prepare_moment_blocks(moment_blocks, n_features):
    if not moment_blocks:
        raise ValueError("moment_blocks must contain at least one block")

    prepared = []
    group_indices = {}
    positive_weight_by_group = {}

    for block_index, source in enumerate(moment_blocks):
        if not isinstance(source, Mapping):
            raise TypeError("each moment block must be a mapping")

        order = int(source["moment_order"])
        if order not in (0, 1, 2):
            raise ValueError("moment_order must be 0, 1, or 2")
        moment_family = source.get("moment_family", order)
        if moment_family == RADIAL_SECOND_MOMENT:
            if order != 2:
                raise ValueError("radial_second blocks must have moment_order=2")
        elif moment_family != order:
            raise ValueError("unsupported moment_family")
        compensated = bool(source.get("compensated", False))
        if compensated and moment_family != RADIAL_SECOND_MOMENT:
            raise ValueError("compensation is only defined for radial moments")

        centers = np.asarray(source["test_centers"], dtype=float)
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("every test_centers block must be non-empty and 2D")
        if centers.shape[1] != n_features:
            raise ValueError("moment-block and mean dimensions do not match")

        directions = source.get("test_directions")
        if directions is None and (order == 0 or moment_family == RADIAL_SECOND_MOMENT):
            directions = np.zeros_like(centers)
        directions = np.asarray(directions, dtype=float)
        if directions.shape != centers.shape:
            raise ValueError("test_directions must match test_centers")

        second_directions = source.get("second_test_directions")
        if order == 2:
            if second_directions is None:
                second_directions = directions
            second_directions = np.asarray(second_directions, dtype=float)
            if second_directions.shape != centers.shape:
                raise ValueError(
                    "second_test_directions must match second-order centers"
                )
        else:
            second_directions = None

        observed = np.asarray(source["Z"], dtype=float).reshape(-1)
        if observed.shape != (centers.shape[0],):
            raise ValueError("Z must have one value per test center")
        if not np.all(np.isfinite(observed)):
            raise ValueError("Z must contain only finite values")

        s = float(source["s"])
        if not np.isfinite(s) or s <= 0:
            raise ValueError("each bandwidth s must be finite and positive")

        moment_scale = float(source.get("moment_scale", 1.0))
        if not np.isfinite(moment_scale) or moment_scale <= 0:
            raise ValueError("moment_scale must be finite and positive")
        observed = moment_scale * observed

        loss_weight = float(source.get("loss_weight", 1.0))
        if not np.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError("loss_weight must be finite and non-negative")
        normalize_loss = bool(source.get("normalize_loss", True))
        denominator = centers.shape[0] if normalize_loss else 1
        residual_scale = np.sqrt(loss_weight / denominator)

        group_label = source.get("amplitude_group", 0)
        try:
            group_index = group_indices[group_label]
        except KeyError:
            group_index = len(group_indices)
            group_indices[group_label] = group_index
            positive_weight_by_group[group_index] = False
        except TypeError as exc:
            raise TypeError("amplitude_group labels must be hashable") from exc
        positive_weight_by_group[group_index] |= residual_scale > 0

        prepared.append(
            {
                "block_index": block_index,
                "moment_order": order,
                "moment_family": moment_family,
                "compensated": compensated,
                "test_centers": centers,
                "test_directions": directions,
                "second_test_directions": second_directions,
                "Z": observed,
                "s": s,
                "moment_scale": moment_scale,
                "loss_weight": loss_weight,
                "normalize_loss": normalize_loss,
                "residual_scale": residual_scale,
                "group_index": group_index,
                "group_label": group_label,
            }
        )

    empty_groups = [
        index
        for index, has_weight in positive_weight_by_group.items()
        if not has_weight
    ]
    if empty_groups:
        raise ValueError("every amplitude group needs a positive-weight moment block")

    group_labels = list(group_indices)
    return prepared, group_labels


def fit_dimension_free_moment_gmm(
    moment_blocks,
    means_init,
    sigmas_init,
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
    amplitude_optimization="non_negative",
    geometry_optimization="component",
    verbose=False,
    iteration_callback=None,
):
    """Fit arbitrary zeroth-, first-, and second-order moment blocks.

    Each block is a mapping with ``Z``, ``test_centers``, ``s``, and
    ``moment_order``.  First- and second-order blocks also use
    ``test_directions``; second-order blocks optionally use
    ``second_test_directions`` (the first directions are reused when omitted).
    A second-order block with ``moment_family='radial_second'`` instead uses
    the normalized radial moment; ``compensated=True`` subtracts its
    zeroth-order kernel part.

    Blocks may provide a positive ``moment_scale``.  It multiplies both the
    observed response ``Z`` and the corresponding analytical response; the
    moment model uses ``s**(-moment_order)`` to obtain dimensionless test
    functions.  Blocks with the same ``amplitude_group`` share an amplitude
    vector.  With
    ``amplitude_optimization='non_negative'``, every group is fitted
    independently by NNLS.  With ``amplitude_optimization='simplex'``, one
    common vector of mixture weights is fitted under ``weights >= 0`` and
    ``sum(weights) == 1``; scale-specific dimension-free amplitudes are then
    derived from those weights and the current sigmas.  A block contributes

    ``0.5 * loss_weight / J * ||Z - Q @ amplitudes||**2``

    when ``normalize_loss=True`` (the default), and omits ``/ J`` otherwise.
    Geometry can be updated one component at a time or jointly. Optional
    joint-polish sweeps start from the converged outer-loop solution and test
    whether coupled component moves can decrease the same objective further.
    """
    means = np.asarray(means_init, dtype=float).copy()
    sigmas = np.asarray(sigmas_init, dtype=float).reshape(-1).copy()
    if means.ndim != 2 or means.shape[0] == 0:
        raise ValueError("means_init must be a non-empty two-dimensional array")
    K, d = means.shape
    if sigmas.shape != (K,) or not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas_init must be finite and match means_init")
    sigmas = np.maximum(sigmas, MIN_SIGMA)
    if not np.all(np.isfinite(means)):
        raise ValueError("means_init must contain only finite values")

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
    if amplitude_optimization == "simplex":
        for group, indices in enumerate(block_indices_per_group):
            if not all(
                np.isclose(blocks[index]["s"], group_bandwidths[group])
                for index in indices
            ):
                raise ValueError(
                    "simplex optimization requires one bandwidth per amplitude group"
                )

    if means_reference is None:
        means_reference = means.copy()
    else:
        means_reference = np.asarray(means_reference, dtype=float).copy()
    if means_reference.shape != means.shape:
        raise ValueError("means_reference must match means_init")

    mean_penalty_weight = float(mean_penalty_weight)
    if not np.isfinite(mean_penalty_weight) or mean_penalty_weight < 0:
        raise ValueError("mean_penalty_weight must be finite and non-negative")
    if geometry_optimization not in ("component", "joint"):
        raise ValueError("geometry_optimization must be 'component' or 'joint'")
    if amplitude_optimization not in ("non_negative", "simplex"):
        raise ValueError("amplitude_optimization must be 'non_negative' or 'simplex'")
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
    if int(joint_polish_sweeps) < 0:
        raise ValueError("joint_polish_sweeps must be non-negative")
    if joint_polish_max_nfev is not None and int(joint_polish_max_nfev) < 1:
        raise ValueError("joint_polish_max_nfev must be positive or None")

    has_explicit_amplitudes_init = amplitudes_init is not None
    if amplitudes_init is None:
        amplitudes = np.full((n_groups, K), 1.0 / K, dtype=float)
    else:
        amplitudes = np.asarray(amplitudes_init, dtype=float).copy()
        if amplitudes.ndim == 1:
            if amplitudes.shape != (K,):
                raise ValueError("amplitudes_init must match the component count")
            amplitudes = np.tile(amplitudes[None, :], (n_groups, 1))
        if amplitudes.shape != (n_groups, K):
            raise ValueError(
                "amplitudes_init must have shape (n_amplitude_groups, n_components)"
            )
        if np.any(amplitudes < 0) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("amplitudes_init must be finite and non-negative")

    if weights_init is None:
        mixture_weights = None
    else:
        mixture_weights = np.asarray(weights_init, dtype=float).reshape(-1).copy()
        if mixture_weights.shape != (K,):
            raise ValueError("weights_init must match the component count")
        if np.any(mixture_weights < 0) or not np.all(np.isfinite(mixture_weights)):
            raise ValueError("weights_init must be finite and non-negative")
        weight_sum = np.sum(mixture_weights)
        if weight_sum <= np.finfo(float).tiny:
            raise ValueError("weights_init must have a positive sum")
        mixture_weights /= weight_sum

    sqrt_penalty_weight = np.sqrt(mean_penalty_weight)

    def evaluate_block_Q_and_jacobian(
        block,
        means_,
        variances_,
        *,
        compute_jacobian,
    ):
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
        moment_scale = block["moment_scale"]
        if moment_scale != 1.0:
            Q = moment_scale * Q
            if jacobian is not None:
                jacobian = moment_scale * jacobian
        return Q, jacobian

    def build_Q_list(means_, sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, MIN_SIGMA * MIN_SIGMA)
        return [
            evaluate_block_Q_and_jacobian(
                block,
                means_,
                variances_,
                compute_jacobian=False,
            )[0]
            for block in blocks
        ]

    def amplitude_factors(sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, MIN_SIGMA * MIN_SIGMA)
        return np.exp(
            -0.5 * d * np.log1p(variances_[None, :] / (group_bandwidths[:, None] ** 2))
        )

    def amplitude_log_eta_derivatives(sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, MIN_SIGMA * MIN_SIGMA)
        s2 = group_bandwidths[:, None] ** 2
        return -0.5 * d * variances_[None, :] / (s2 + variances_[None, :])

    def amplitudes_from_weights(weights_, sigmas_):
        return amplitude_factors(sigmas_) * weights_[None, :]

    if amplitude_optimization == "simplex" and mixture_weights is None:
        if has_explicit_amplitudes_init:
            raw_weights = amplitudes / amplitude_factors(sigmas)
            row_sums = np.sum(raw_weights, axis=1, keepdims=True)
            valid_rows = row_sums[:, 0] > np.finfo(float).tiny
            if np.any(valid_rows):
                normalized_rows = raw_weights[valid_rows] / row_sums[valid_rows]
                mixture_weights = np.mean(normalized_rows, axis=0)
                mixture_weights /= np.sum(mixture_weights)
            else:
                mixture_weights = np.full(K, 1.0 / K, dtype=float)
        else:
            mixture_weights = np.full(K, 1.0 / K, dtype=float)

    def solve_amplitudes(Q_list, sigmas_, weights_start=None):
        if amplitude_optimization == "non_negative":
            solved = np.empty((n_groups, K), dtype=float)
            for group, indices in enumerate(block_indices_per_group):
                design = np.vstack(
                    [blocks[i]["residual_scale"] * Q_list[i] for i in indices]
                )
                target = np.concatenate(
                    [blocks[i]["residual_scale"] * blocks[i]["Z"] for i in indices]
                )
                solved[group] = nnls(design, target)[0]
            return solved, None

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
            solved_weights = np.ones(1, dtype=float)
        else:
            if weights_start is None:
                initial_weights = np.full(K, 1.0 / K, dtype=float)
            else:
                initial_weights = np.asarray(weights_start, dtype=float).copy()
                initial_weights = np.maximum(initial_weights, 0.0)
                initial_weights /= np.sum(initial_weights)

            def simplex_objective(weights_):
                residual = design @ weights_ - target
                return 0.5 * np.dot(residual, residual)

            def simplex_jacobian(weights_):
                return design.T @ (design @ weights_ - target)

            result = minimize(
                simplex_objective,
                initial_weights,
                jac=simplex_jacobian,
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
            solved_weights = np.clip(result.x, 0.0, 1.0)
            solved_weights /= np.sum(solved_weights)

        return amplitudes_from_weights(solved_weights, sigmas_), solved_weights

    def data_objective(Q_list, amplitudes_):
        total = 0.0
        for block, Q in zip(blocks, Q_list):
            a = amplitudes_[block["group_index"]]
            residual = block["Z"] - Q @ a
            total += (
                0.5
                * block["residual_scale"] ** 2
                * np.dot(
                    residual,
                    residual,
                )
            )
        return total

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

    def penalty_objective(means_):
        delta = means_ - means_reference
        return 0.5 * mean_penalty_weight * np.sum(delta * delta)

    def objective(Q_list, amplitudes_, means_):
        return data_objective(Q_list, amplitudes_) + penalty_objective(means_)

    def geometry_bounds(means_, sigmas_):
        theta = _pack_geometry(means_, sigmas_)
        component_theta = theta.reshape(K, d + 1)
        lower = np.full((K, d + 1), -np.inf)
        upper = np.full((K, d + 1), np.inf)
        if eta_step_bound is not None:
            lower[:, -1] = component_theta[:, -1] - eta_step_bound
            upper[:, -1] = component_theta[:, -1] + eta_step_bound
        if m_step_bound is not None:
            lower[:, :d] = component_theta[:, :d] - m_step_bound
            upper[:, :d] = component_theta[:, :d] + m_step_bound
        return theta, lower, upper

    penalty_jacobian = None
    if mean_penalty_weight > 0:
        penalty_jacobian = _build_mean_penalty_jacobian(
            K,
            d,
            sqrt_penalty_weight,
        )

    geometry_diagnostics = {
        "calls": 0,
        "nfev": 0,
        "njev": 0,
        "max_nfev_hits": 0,
        "accepted": 0,
        "eta_bound_hits": 0,
        "eta_coordinates": 0,
        "optimality_sum": 0.0,
        "optimality_max": 0.0,
        "joint_polish_calls": 0,
        "joint_polish_nfev": 0,
    }

    def record_geometry_result(result, accepted, n_eta_coordinates, *, polish):
        geometry_diagnostics["calls"] += 1
        geometry_diagnostics["nfev"] += int(result.nfev)
        geometry_diagnostics["njev"] += int(result.njev or 0)
        geometry_diagnostics["max_nfev_hits"] += int(result.status == 0)
        geometry_diagnostics["accepted"] += int(accepted)
        active_mask = np.asarray(result.active_mask, dtype=int).reshape(
            -1,
            d + 1,
        )
        geometry_diagnostics["eta_bound_hits"] += int(
            np.count_nonzero(active_mask[:, -1])
        )
        geometry_diagnostics["eta_coordinates"] += int(n_eta_coordinates)
        optimality = float(result.optimality)
        geometry_diagnostics["optimality_sum"] += optimality
        geometry_diagnostics["optimality_max"] = max(
            geometry_diagnostics["optimality_max"],
            optimality,
        )
        if polish:
            geometry_diagnostics["joint_polish_calls"] += 1
            geometry_diagnostics["joint_polish_nfev"] += int(result.nfev)

    def joint_geometry_sweep(
        Q_list,
        amplitudes_,
        mixture_weights_,
        *,
        max_nfev=None,
        polish=False,
    ):
        nonlocal means, sigmas
        theta_start, lower, upper = geometry_bounds(means, sigmas)
        cache = {"theta": None}

        def evaluate(theta):
            if cache["theta"] is not None and np.array_equal(theta, cache["theta"]):
                return cache["value"]

            means_, sigmas_, variances_ = _unpack_geometry(theta, K, d)
            if amplitude_optimization == "simplex":
                candidate_amplitudes = amplitudes_from_weights(
                    mixture_weights_,
                    sigmas_,
                )
                amplitude_eta_derivatives = (
                    candidate_amplitudes * amplitude_log_eta_derivatives(sigmas_)
                )
            else:
                candidate_amplitudes = amplitudes_
                amplitude_eta_derivatives = None
            Q_new_list = []
            jacobians = []
            for block in blocks:
                Q, jac_q = evaluate_block_Q_and_jacobian(
                    block,
                    means_,
                    variances_,
                    compute_jacobian=True,
                )
                Q_new_list.append(Q)
                group = block["group_index"]
                a = candidate_amplitudes[group]
                prediction_jacobian = jac_q * a[None, :, None]
                if amplitude_eta_derivatives is not None:
                    prediction_jacobian[:, :, -1] += (
                        Q * amplitude_eta_derivatives[group][None, :]
                    )
                jacobians.append(
                    -block["residual_scale"]
                    * prediction_jacobian.reshape(-1, K * (d + 1))
                )
            value = (
                means_,
                sigmas_,
                Q_new_list,
                candidate_amplitudes,
                np.vstack(jacobians),
            )
            cache["theta"] = theta.copy()
            cache["value"] = value
            return value

        def residual(theta):
            means_, _, Q_new_list, candidate_amplitudes, _ = evaluate(theta)
            data_residuals = [
                block["residual_scale"]
                * (block["Z"] - Q @ candidate_amplitudes[block["group_index"]])
                for block, Q in zip(blocks, Q_new_list)
            ]
            if mean_penalty_weight > 0:
                data_residuals.append(
                    sqrt_penalty_weight * (means_ - means_reference).reshape(-1)
                )
            return np.concatenate(data_residuals)

        def residual_jacobian(theta):
            data_jacobian = evaluate(theta)[4]
            if mean_penalty_weight == 0:
                return data_jacobian
            return np.vstack([data_jacobian, penalty_jacobian])

        result = least_squares(
            residual,
            theta_start,
            jac=residual_jacobian,
            bounds=(lower.reshape(-1), upper.reshape(-1)),
            method="trf",
            max_nfev=int(ls_max_nfev if max_nfev is None else max_nfev),
        )
        means_new, sigmas_new, Q_new_list, candidate_amplitudes, _ = evaluate(result.x)
        accepted = objective(Q_new_list, candidate_amplitudes, means_new) <= objective(
            Q_list,
            amplitudes_,
            means,
        )
        record_geometry_result(result, accepted, K, polish=polish)
        if accepted:
            means = means_new.copy()
            sigmas = sigmas_new.copy()
            return [Q.copy() for Q in Q_new_list], candidate_amplitudes.copy()
        return Q_list, amplitudes_

    def component_geometry_sweep(Q_list, amplitudes_, mixture_weights_):
        nonlocal means, sigmas
        max_amplitude = float(np.max(amplitudes_))
        if max_amplitude <= 0:
            return Q_list, amplitudes_

        for component in range(K):
            component_amplitudes = amplitudes_[:, component]
            if np.max(component_amplitudes) <= active_tol * max_amplitude:
                continue

            targets = []
            for block, Q in zip(blocks, Q_list):
                a = amplitudes_[block["group_index"]]
                targets.append(block["Z"] - Q @ a + a[component] * Q[:, component])

            mean_start = means[component].copy()
            eta_start = np.log(max(sigmas[component], MIN_SIGMA) ** 2)
            theta_start = np.concatenate([mean_start, [eta_start]])
            lower = np.full(d + 1, -np.inf)
            upper = np.full(d + 1, np.inf)
            if eta_step_bound is not None:
                lower[-1] = eta_start - eta_step_bound
                upper[-1] = eta_start + eta_step_bound
            if m_step_bound is not None:
                lower[:d] = mean_start - m_step_bound
                upper[:d] = mean_start + m_step_bound

            cache = {"theta": None}

            def evaluate(theta):
                if cache["theta"] is not None and np.array_equal(
                    theta,
                    cache["theta"],
                ):
                    return cache["value"]

                mean = theta[:d]
                variance = max(float(np.exp(theta[-1])), MIN_SIGMA * MIN_SIGMA)
                if amplitude_optimization == "simplex":
                    factors = np.exp(
                        -0.5
                        * d
                        * np.log1p(variance / (group_bandwidths * group_bandwidths))
                    )
                    candidate_component_amplitudes = (
                        mixture_weights_[component] * factors
                    )
                    candidate_amplitude_eta_derivatives = (
                        candidate_component_amplitudes
                        * (-0.5 * d * variance)
                        / (group_bandwidths * group_bandwidths + variance)
                    )
                else:
                    candidate_component_amplitudes = amplitudes_[:, component]
                    candidate_amplitude_eta_derivatives = np.zeros(
                        n_groups,
                        dtype=float,
                    )
                q_values = []
                q_jacobians = []
                for block in blocks:
                    q, jac_q = evaluate_block_Q_and_jacobian(
                        block,
                        mean[None, :],
                        np.array([variance]),
                        compute_jacobian=True,
                    )
                    q_values.append(q[:, 0])
                    q_jacobians.append(jac_q[:, 0, :])
                value = (
                    mean,
                    np.sqrt(variance),
                    q_values,
                    q_jacobians,
                    candidate_component_amplitudes,
                    candidate_amplitude_eta_derivatives,
                )
                cache["theta"] = theta.copy()
                cache["value"] = value
                return value

            def residual(theta):
                mean, _, q_values, _, candidate_amplitudes, _ = evaluate(theta)
                pieces = []
                for block, target, q in zip(blocks, targets, q_values):
                    amplitude = candidate_amplitudes[block["group_index"]]
                    pieces.append(block["residual_scale"] * (target - amplitude * q))
                if mean_penalty_weight > 0:
                    pieces.append(
                        sqrt_penalty_weight * (mean - means_reference[component])
                    )
                return np.concatenate(pieces)

            def residual_jacobian(theta):
                _, _, q_values, q_jacobians, candidate_amplitudes, derivatives = (
                    evaluate(theta)
                )
                pieces = []
                for block, q, jac_q in zip(blocks, q_values, q_jacobians):
                    group = block["group_index"]
                    prediction_jacobian = candidate_amplitudes[group] * jac_q.copy()
                    prediction_jacobian[:, -1] += q * derivatives[group]
                    pieces.append(-block["residual_scale"] * prediction_jacobian)
                if mean_penalty_weight > 0:
                    penalty = np.zeros((d, d + 1), dtype=float)
                    penalty[:, :d] = sqrt_penalty_weight * np.eye(d)
                    pieces.append(penalty)
                return np.vstack(pieces)

            result = least_squares(
                residual,
                theta_start,
                jac=residual_jacobian,
                bounds=(lower, upper),
                method="trf",
                max_nfev=int(ls_max_nfev),
            )
            (
                mean_new,
                sigma_new,
                q_new_values,
                _,
                candidate_component_amplitudes,
                _,
            ) = evaluate(result.x)

            old_local = 0.0
            new_local = 0.0
            for block, target, Q, q_new in zip(
                blocks,
                targets,
                Q_list,
                q_new_values,
            ):
                group = block["group_index"]
                old_amplitude = amplitudes_[group, component]
                new_amplitude = candidate_component_amplitudes[group]
                old_residual = target - old_amplitude * Q[:, component]
                new_residual = target - new_amplitude * q_new
                scale2 = block["residual_scale"] ** 2
                old_local += 0.5 * scale2 * np.dot(old_residual, old_residual)
                new_local += 0.5 * scale2 * np.dot(new_residual, new_residual)
            if mean_penalty_weight > 0:
                old_delta = means[component] - means_reference[component]
                new_delta = mean_new - means_reference[component]
                old_local += (
                    0.5
                    * mean_penalty_weight
                    * np.dot(
                        old_delta,
                        old_delta,
                    )
                )
                new_local += (
                    0.5
                    * mean_penalty_weight
                    * np.dot(
                        new_delta,
                        new_delta,
                    )
                )

            accepted = new_local <= old_local
            record_geometry_result(result, accepted, 1, polish=False)
            if accepted:
                means[component] = mean_new
                sigmas[component] = sigma_new
                amplitudes_[:, component] = candidate_component_amplitudes
                for Q, q_new in zip(Q_list, q_new_values):
                    Q[:, component] = q_new

        return Q_list, amplitudes_

    def final_geometry_gradient_norms(amplitudes_, mixture_weights_):
        variances_ = np.maximum(sigmas * sigmas, MIN_SIGMA * MIN_SIGMA)
        if amplitude_optimization == "simplex":
            final_amplitudes = amplitudes_from_weights(mixture_weights_, sigmas)
            amplitude_eta_derivatives = (
                final_amplitudes * amplitude_log_eta_derivatives(sigmas)
            )
        else:
            final_amplitudes = amplitudes_
            amplitude_eta_derivatives = None

        gradient = np.zeros(K * (d + 1), dtype=float)
        for block in blocks:
            Q, jac_q = evaluate_block_Q_and_jacobian(
                block,
                means,
                variances_,
                compute_jacobian=True,
            )
            group = block["group_index"]
            a = final_amplitudes[group]
            prediction_jacobian = jac_q * a[None, :, None]
            if amplitude_eta_derivatives is not None:
                prediction_jacobian[:, :, -1] += (
                    Q * amplitude_eta_derivatives[group][None, :]
                )
            residual = block["residual_scale"] * (block["Z"] - Q @ a)
            residual_jacobian = -block["residual_scale"] * prediction_jacobian.reshape(
                -1, K * (d + 1)
            )
            gradient += residual_jacobian.T @ residual

        if mean_penalty_weight > 0:
            penalty_residual = sqrt_penalty_weight * (means - means_reference).reshape(
                -1
            )
            gradient += penalty_jacobian.T @ penalty_residual

        component_gradient = gradient.reshape(K, d + 1)
        return (
            float(np.max(np.abs(component_gradient[:, :d]))),
            float(np.max(np.abs(component_gradient[:, -1]))),
        )

    def current_mixture_weights(amplitudes_, mixture_weights_):
        if mixture_weights_ is not None:
            return np.asarray(mixture_weights_, dtype=float).copy()
        raw_weights = amplitudes_ / np.maximum(
            amplitude_factors(sigmas),
            np.finfo(float).tiny,
        )
        row_sums = np.sum(raw_weights, axis=1, keepdims=True)
        valid = row_sums[:, 0] > np.finfo(float).tiny
        if not np.any(valid):
            return np.full(K, 1.0 / K, dtype=float)
        weights = np.mean(raw_weights[valid] / row_sums[valid], axis=0)
        return weights / np.sum(weights)

    def report_iteration(
        iteration,
        current_objective,
        current_data_objective,
        current_penalty_objective,
        *,
        phase,
    ):
        if iteration_callback is None:
            return
        iteration_callback(
            IterationSnapshot(
                iteration=iteration,
                loss=current_objective,
                parameters=MixtureParameters.spherical(
                    means,
                    current_mixture_weights(amplitudes, mixture_weights),
                    sigmas,
                ),
                phase=phase,
                losses={
                    "data": current_data_objective,
                    "penalty": current_penalty_objective,
                },
            )
        )

    start_time = time.time()
    Q_list = build_Q_list(means, sigmas)
    amplitudes, mixture_weights = solve_amplitudes(
        Q_list,
        sigmas,
        mixture_weights,
    )
    previous_objective = objective(Q_list, amplitudes, means)
    initial_objective = previous_objective
    stable_iterations = 0
    history = []
    data_history = []
    penalty_history = []
    report_iteration(
        0,
        initial_objective,
        data_objective(Q_list, amplitudes),
        penalty_objective(means),
        phase="initialization",
    )

    iterator = range(int(n_outer))
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator)

    for outer_iteration in iterator:
        for _ in range(int(geom_sweeps)):
            if geometry_optimization == "joint":
                Q_list, amplitudes = joint_geometry_sweep(
                    Q_list,
                    amplitudes,
                    mixture_weights,
                )
            else:
                Q_list, amplitudes = component_geometry_sweep(
                    Q_list,
                    amplitudes,
                    mixture_weights,
                )

        amplitudes, mixture_weights = solve_amplitudes(
            Q_list,
            sigmas,
            mixture_weights,
        )
        current_data_objective = data_objective(Q_list, amplitudes)
        current_penalty_objective = penalty_objective(means)
        current_objective = current_data_objective + current_penalty_objective
        history.append(current_objective)
        data_history.append(current_data_objective)
        penalty_history.append(current_penalty_objective)
        report_iteration(
            outer_iteration + 1,
            current_objective,
            current_data_objective,
            current_penalty_objective,
            phase="optimization",
        )

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

    n_outer_iter = len(history)
    converged = stable_iterations >= convergence_patience
    pre_polish_objective = objective(Q_list, amplitudes, means)
    polish_history = []
    for polish_iteration in range(int(joint_polish_sweeps)):
        Q_list, amplitudes = joint_geometry_sweep(
            Q_list,
            amplitudes,
            mixture_weights,
            max_nfev=joint_polish_max_nfev,
            polish=True,
        )
        amplitudes, mixture_weights = solve_amplitudes(
            Q_list,
            sigmas,
            mixture_weights,
        )
        polish_objective = objective(Q_list, amplitudes, means)
        polish_history.append(polish_objective)
        report_iteration(
            n_outer_iter + polish_iteration + 1,
            polish_objective,
            data_objective(Q_list, amplitudes),
            penalty_objective(means),
            phase="polish",
        )

    elapsed = time.time() - start_time
    final_data_objective = data_objective(Q_list, amplitudes)
    final_penalty_objective = penalty_objective(means)
    final_objective = final_data_objective + final_penalty_objective
    final_mean_gradient_inf, final_eta_gradient_inf = final_geometry_gradient_norms(
        amplitudes, mixture_weights
    )
    objective_scale = max(abs(pre_polish_objective), np.finfo(float).tiny)
    joint_polish_relative_gain = (
        pre_polish_objective - final_objective
    ) / objective_scale
    geometry_calls = geometry_diagnostics["calls"]
    eta_coordinates = geometry_diagnostics["eta_coordinates"]
    if verbose:
        print(f"fit_dimension_free_moment_gmm finished in {elapsed:.3f} seconds")

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
        "polish_history": np.asarray(polish_history),
        "data_objective": final_data_objective,
        "data_history": np.asarray(data_history),
        "data_objective_per_order": data_objective_per_order(Q_list, amplitudes),
        "data_objective_per_family": data_objective_per_family(Q_list, amplitudes),
        "penalty_objective": final_penalty_objective,
        "penalty_history": np.asarray(penalty_history),
        "n_iter": n_outer_iter,
        "n_outer_iter": n_outer_iter,
        "converged": converged,
        "geometry_calls": geometry_calls,
        "geometry_nfev": geometry_diagnostics["nfev"],
        "geometry_njev": geometry_diagnostics["njev"],
        "geometry_max_nfev_hits": geometry_diagnostics["max_nfev_hits"],
        "geometry_accepted": geometry_diagnostics["accepted"],
        "geometry_eta_bound_hits": geometry_diagnostics["eta_bound_hits"],
        "geometry_eta_coordinates": eta_coordinates,
        "geometry_max_nfev_fraction": (
            geometry_diagnostics["max_nfev_hits"] / geometry_calls
            if geometry_calls
            else 0.0
        ),
        "geometry_acceptance_rate": (
            geometry_diagnostics["accepted"] / geometry_calls if geometry_calls else 0.0
        ),
        "geometry_eta_bound_fraction": (
            geometry_diagnostics["eta_bound_hits"] / eta_coordinates
            if eta_coordinates
            else 0.0
        ),
        "geometry_mean_optimality": (
            geometry_diagnostics["optimality_sum"] / geometry_calls
            if geometry_calls
            else 0.0
        ),
        "geometry_max_optimality": geometry_diagnostics["optimality_max"],
        "joint_polish_calls": geometry_diagnostics["joint_polish_calls"],
        "joint_polish_nfev": geometry_diagnostics["joint_polish_nfev"],
        "pre_polish_objective": pre_polish_objective,
        "joint_polish_relative_gain": joint_polish_relative_gain,
        "final_mean_gradient_inf": final_mean_gradient_inf,
        "final_eta_gradient_inf": final_eta_gradient_inf,
        "elapsed_seconds": elapsed,
        "loss_scales": np.asarray(
            [block["residual_scale"] for block in blocks],
            dtype=float,
        ),
    }
