import time

import numpy as np
from scipy.optimize import least_squares, nnls

from .utils import (
    MIN_SIGMA,
    _build_mean_penalty_jacobian,
    _evaluate_dimension_free_Q_and_jacobian,
    _pack_geometry,
    _unpack_geometry,
)


def fit_dimension_free_multi_s_gmm(
    Z_list,
    test_centers,
    test_directions,
    s_values,
    means_init,
    sigmas_init,
    amplitudes_init=None,
    n_outer=50,
    geom_sweeps=1,
    ls_max_nfev=30,
    active_tol=1e-8,
    eta_step_bound=0.75,
    m_step_bound=None,
    means_reference=None,
    mean_penalty_weight=0.0,
    objective_rtol=1e-6,
    objective_atol=0.0,
    convergence_patience=2,
    verbose=False,
):
    """Fit multiple bandwidths with shared geometry and scale-specific amplitudes.

    For each bandwidth ``s_l`` the amplitude vector ``a_l`` is solved by NNLS,
    while all bandwidths share the same component means and sigmas.
    """
    X = np.asarray(test_centers, dtype=float)
    U = np.asarray(test_directions, dtype=float)
    s_values = np.asarray(s_values, dtype=float).reshape(-1)
    means = np.asarray(means_init, dtype=float).copy()
    sigmas = np.asarray(sigmas_init, dtype=float).reshape(-1).copy()
    sigmas = np.maximum(sigmas, MIN_SIGMA)
    Z_list = [np.asarray(Z, dtype=float).reshape(-1) for Z in Z_list]

    if means_reference is None:
        means_reference = means.copy()
    else:
        means_reference = np.asarray(means_reference, dtype=float).copy()

    mean_penalty_weight = float(mean_penalty_weight)

    assert X.ndim == 2, "test_centers must be 2D"
    assert U.shape == X.shape, "test_directions must match test_centers"
    assert means.ndim == 2, "means must be 2D"
    assert means.shape[1] == X.shape[1], "means dimension mismatch"
    assert sigmas.shape == (means.shape[0],), "sigmas length must match means"
    assert s_values.ndim == 1 and s_values.size > 0, "s_values must be non-empty"
    assert np.all(np.isfinite(s_values)) and np.all(s_values > 0)
    assert len(Z_list) == s_values.size, "Z_list length must match s_values"
    assert all(Z.shape == (X.shape[0],) for Z in Z_list)
    assert np.all(np.isfinite(sigmas)) and np.all(sigmas > 0)
    assert means_reference.shape == means.shape
    assert mean_penalty_weight >= 0
    assert active_tol >= 0
    assert objective_rtol >= 0
    assert objective_atol >= 0
    assert convergence_patience >= 1

    t0 = time.time()
    _, d = X.shape
    K = means.shape[0]
    L = s_values.size
    sqrt_penalty_weight = np.sqrt(mean_penalty_weight)

    if amplitudes_init is None:
        amplitudes = np.full((L, K), 1.0 / K, dtype=float)
    else:
        amplitudes = np.asarray(amplitudes_init, dtype=float).copy()
        if amplitudes.ndim == 1:
            amplitudes = np.tile(amplitudes.reshape(1, K), (L, 1))
        assert amplitudes.shape == (L, K)

    def build_Q_list(means_, sigmas_):
        variances_ = np.maximum(sigmas_ * sigmas_, MIN_SIGMA * MIN_SIGMA)
        return [
            _evaluate_dimension_free_Q_and_jacobian(
                X,
                U,
                means_,
                variances_,
                s,
                compute_jacobian=False,
            )[0]
            for s in s_values
        ]

    def solve_amplitudes(Q_list):
        return np.vstack([nnls(Q, Z)[0] for Q, Z in zip(Q_list, Z_list)])

    def data_objective(Q_list, amplitudes_):
        total = 0.0
        for Q, Z, a in zip(Q_list, Z_list, amplitudes_):
            residual = Z - Q @ a
            total += 0.5 * np.dot(residual, residual)
        return total

    def penalty_objective(means_):
        delta = means_ - means_reference
        return 0.5 * mean_penalty_weight * np.sum(delta * delta)

    def objective(Q_list, amplitudes_, means_):
        return data_objective(Q_list, amplitudes_) + penalty_objective(means_)

    penalty_jacobian = None
    if mean_penalty_weight > 0:
        penalty_jacobian = _build_mean_penalty_jacobian(K, d, sqrt_penalty_weight)

    history = []
    data_history = []
    penalty_history = []

    iterator = range(n_outer)
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator)

    Q_list = build_Q_list(means, sigmas)
    amplitudes = solve_amplitudes(Q_list)
    prev_obj = objective(Q_list, amplitudes, means)
    stable_iterations = 0

    for _ in iterator:
        amplitudes = solve_amplitudes(Q_list)

        for _ in range(geom_sweeps):
            theta_start = _pack_geometry(means, sigmas)
            component_start = theta_start.reshape(K, d + 1)

            lower = np.full((K, d + 1), -np.inf)
            upper = np.full((K, d + 1), np.inf)
            if eta_step_bound is not None:
                lower[:, -1] = component_start[:, -1] - eta_step_bound
                upper[:, -1] = component_start[:, -1] + eta_step_bound
            if m_step_bound is not None:
                lower[:, :d] = component_start[:, :d] - m_step_bound
                upper[:, :d] = component_start[:, :d] + m_step_bound

            evaluation_cache = {
                "theta": None,
                "means": None,
                "sigmas": None,
                "Q_list": None,
                "data_jacobian": None,
            }

            def evaluate_model_and_jacobian(theta):
                cached_theta = evaluation_cache["theta"]
                if cached_theta is not None and np.array_equal(theta, cached_theta):
                    return (
                        evaluation_cache["means"],
                        evaluation_cache["sigmas"],
                        evaluation_cache["Q_list"],
                        evaluation_cache["data_jacobian"],
                    )

                means_, sigmas_, variances_ = _unpack_geometry(theta, K, d)

                Q_new_list = []
                data_jacobians = []
                for s, a in zip(s_values, amplitudes):
                    Q_, jac_q = _evaluate_dimension_free_Q_and_jacobian(
                        X,
                        U,
                        means_,
                        variances_,
                        s,
                        compute_jacobian=True,
                    )
                    Q_new_list.append(Q_)
                    data_jacobians.append(
                        -(jac_q * a[None, :, None]).reshape(X.shape[0], K * (d + 1))
                    )
                data_jacobian = np.vstack(data_jacobians)

                evaluation_cache["theta"] = theta.copy()
                evaluation_cache["means"] = means_
                evaluation_cache["sigmas"] = sigmas_
                evaluation_cache["Q_list"] = Q_new_list
                evaluation_cache["data_jacobian"] = data_jacobian
                return means_, sigmas_, Q_new_list, data_jacobian

            def residual(theta):
                means_, _, Q_new_list, _ = evaluate_model_and_jacobian(theta)
                data_residuals = [
                    Z - Q @ a for Q, Z, a in zip(Q_new_list, Z_list, amplitudes)
                ]
                if mean_penalty_weight == 0:
                    return np.concatenate(data_residuals)
                penalty_residual = sqrt_penalty_weight * (
                    means_ - means_reference
                ).reshape(-1)
                return np.concatenate(data_residuals + [penalty_residual])

            def residual_jacobian(theta):
                _, _, _, data_jacobian = evaluate_model_and_jacobian(theta)
                if mean_penalty_weight == 0:
                    return data_jacobian
                return np.vstack([data_jacobian, penalty_jacobian])

            res = least_squares(
                residual,
                theta_start,
                jac=residual_jacobian,
                bounds=(lower.reshape(-1), upper.reshape(-1)),
                method="trf",
                max_nfev=ls_max_nfev,
            )

            means_new, sigmas_new, Q_new_list, _ = evaluate_model_and_jacobian(res.x)
            old_local = objective(Q_list, amplitudes, means)
            new_local = objective(Q_new_list, amplitudes, means_new)
            if new_local <= old_local:
                means = means_new.copy()
                sigmas = sigmas_new.copy()
                Q_list = [Q.copy() for Q in Q_new_list]

        amplitudes = solve_amplitudes(Q_list)
        cur_data_obj = data_objective(Q_list, amplitudes)
        cur_penalty_obj = penalty_objective(means)
        cur_obj = cur_data_obj + cur_penalty_obj
        history.append(cur_obj)
        data_history.append(cur_data_obj)
        penalty_history.append(cur_penalty_obj)

        improvement = prev_obj - cur_obj
        tolerance = objective_atol + objective_rtol * max(
            abs(prev_obj), abs(cur_obj), np.finfo(float).tiny
        )
        if improvement >= 0 and improvement <= tolerance:
            stable_iterations += 1
        else:
            stable_iterations = 0

        if stable_iterations >= convergence_patience:
            break
        prev_obj = cur_obj

    elapsed = time.time() - t0
    if verbose:
        print(f"fit_dimension_free_multi_s_gmm finished in {elapsed:.3f} seconds")

    return {
        "means": means,
        "sigmas": sigmas,
        "amplitudes_per_s": amplitudes,
        "amplitudes": amplitudes[-1],
        "objective": history[-1] if history else prev_obj,
        "history": np.asarray(history),
        "data_objective": (
            data_history[-1] if history else data_objective(Q_list, amplitudes)
        ),
        "data_history": np.asarray(data_history),
        "penalty_objective": (
            penalty_history[-1] if history else penalty_objective(means)
        ),
        "penalty_history": np.asarray(penalty_history),
        "n_iter": len(history),
        "elapsed_seconds": elapsed,
    }
