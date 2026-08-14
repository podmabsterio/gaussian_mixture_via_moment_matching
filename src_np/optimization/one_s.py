import time

import numpy as np
from scipy.optimize import least_squares, nnls

from .utils import MIN_SIGMA


def fit_dimension_free_one_s_gmm(
    Z,
    test_centers,
    test_directions,
    s,
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
    Z = np.asarray(Z, dtype=float).reshape(-1)
    X = np.asarray(test_centers, dtype=float)
    U = np.asarray(test_directions, dtype=float)
    means = np.asarray(means_init, dtype=float).copy()
    sigmas = np.asarray(sigmas_init, dtype=float).reshape(-1).copy()
    sigmas = np.maximum(sigmas, MIN_SIGMA)
    amplitudes = (
        None
        if amplitudes_init is None
        else np.asarray(amplitudes_init, dtype=float).reshape(-1).copy()
    )
    if means_reference is None:
        means_reference = means.copy()
    else:
        means_reference = np.asarray(means_reference, dtype=float).copy()
    mean_penalty_weight = float(mean_penalty_weight)

    assert X.ndim == 2, "X must be 2D"
    assert U.shape == X.shape, "U must match X shape"
    assert Z.shape == (X.shape[0],), "Z length must match data points"
    assert means.ndim == 2, "means must be 2D"
    assert means.shape[1] == X.shape[1], "means dimension mismatch"
    assert sigmas.shape == (means.shape[0],), "sigmas length must match means"
    assert np.isscalar(s), "s must be scalar"
    assert s > 0, "s must be positive"
    assert np.all(np.isfinite(sigmas)) and np.all(sigmas > 0), "sigmas must be positive"
    assert amplitudes is None or amplitudes.shape == (
        means.shape[0],
    ), "amplitudes length must match means"
    assert means_reference.shape == means.shape, "means_reference must match means"
    assert mean_penalty_weight >= 0, "mean_penalty_weight must be non-negative"
    assert objective_rtol >= 0, "objective_rtol must be non-negative"
    assert objective_atol >= 0, "objective_atol must be non-negative"
    assert convergence_patience >= 1, "convergence_patience must be at least 1"

    t0 = time.time()
    J, d = X.shape
    K = means.shape[0]
    s2 = float(s) ** 2
    base_mask = np.linalg.norm(U, axis=1) <= np.finfo(float).tiny

    if amplitudes is None:
        amplitudes = np.ones(K) / K

    def q_single(m, sigma):
        sigma = max(float(sigma), MIN_SIGMA)
        r = m[None, :] - X
        t = s2 + sigma**2
        h = np.sum(r * U, axis=1)
        norm2 = np.sum(r * r, axis=1)
        base = np.exp(-0.5 * norm2 / t)
        q = base * (s2 / t) * h
        q[base_mask] = base[base_mask]
        return q

    def build_Q(means_, sigmas_):
        sigmas_ = np.maximum(sigmas_, MIN_SIGMA)
        r = means_[None, :, :] - X[:, None, :]
        t = s2 + sigmas_ * sigmas_
        h = np.einsum("jkd,jd->jk", r, U)
        norm2 = np.einsum("jkd,jkd->jk", r, r)
        base = np.exp(-0.5 * norm2 / t[None, :])
        Q = base * (s2 / t)[None, :] * h
        Q[base_mask] = base[base_mask]
        return Q

    def data_objective(Q, a):
        res = Z - Q @ a
        return 0.5 * np.dot(res, res)

    def penalty_objective(means_):
        delta = means_ - means_reference
        return 0.5 * mean_penalty_weight * np.sum(delta * delta)

    def objective(Q, a, means_):
        return data_objective(Q, a) + penalty_objective(means_)

    history = []
    data_history = []
    penalty_history = []

    iterator = range(n_outer)
    if verbose:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator)

    Q = build_Q(means, sigmas)
    prev_obj = objective(Q, amplitudes, means)
    stable_iterations = 0

    for _ in iterator:
        amplitudes, _ = nnls(Q, Z)

        for _ in range(geom_sweeps):
            pred = Q @ amplitudes
            max_amplitude = np.max(amplitudes)

            for k in range(K):
                a_k = amplitudes[k]
                if max_amplitude <= 0 or a_k <= active_tol * max_amplitude:
                    continue

                q_old = Q[:, k].copy()
                R_minus_k = Z - pred + a_k * q_old

                m_start = means[k].copy()
                eta_start = np.log(sigmas[k] ** 2)
                theta_start = np.concatenate([m_start, [eta_start]])

                lower = np.full(d + 1, -np.inf)
                upper = np.full(d + 1, np.inf)

                if eta_step_bound is not None:
                    lower[-1] = eta_start - eta_step_bound
                    upper[-1] = eta_start + eta_step_bound

                if m_step_bound is not None:
                    lower[:d] = m_start - m_step_bound
                    upper[:d] = m_start + m_step_bound

                evaluation_cache = {
                    "theta": None,
                    "q": None,
                    "jac_q": None,
                }

                def evaluate_q_and_jacobian(theta):
                    cached_theta = evaluation_cache["theta"]
                    if cached_theta is not None and np.array_equal(theta, cached_theta):
                        return evaluation_cache["q"], evaluation_cache["jac_q"]

                    m = theta[:d]
                    variance = np.exp(theta[-1])
                    t = s2 + variance
                    r = m[None, :] - X
                    h = np.sum(r * U, axis=1)
                    norm2 = np.sum(r * r, axis=1)
                    base = np.exp(-0.5 * norm2 / t)
                    scale = base * (s2 / t)
                    q = scale * h
                    q[base_mask] = base[base_mask]

                    jac_q = np.empty((J, d + 1), dtype=float)
                    jac_q[:, :d] = scale[:, None] * (U - (h / t)[:, None] * r)
                    jac_q[:, -1] = q * variance * (-1.0 / t + 0.5 * norm2 / (t * t))
                    if np.any(base_mask):
                        jac_q[base_mask, :d] = (
                            base[base_mask, None] * (-r[base_mask] / t)
                        )
                        jac_q[base_mask, -1] = (
                            base[base_mask]
                            * variance
                            * 0.5
                            * norm2[base_mask]
                            / (t * t)
                        )

                    evaluation_cache["theta"] = theta.copy()
                    evaluation_cache["q"] = q
                    evaluation_cache["jac_q"] = jac_q
                    return q, jac_q

                def residual(theta):
                    q, _ = evaluate_q_and_jacobian(theta)
                    data_residual = R_minus_k - a_k * q
                    if mean_penalty_weight == 0:
                        return data_residual
                    penalty_residual = np.sqrt(mean_penalty_weight) * (
                        theta[:d] - means_reference[k]
                    )
                    return np.concatenate([data_residual, penalty_residual])

                def residual_jacobian(theta):
                    _, jac_q = evaluate_q_and_jacobian(theta)
                    data_jacobian = -a_k * jac_q
                    if mean_penalty_weight == 0:
                        return data_jacobian
                    penalty_jacobian = np.zeros((d, d + 1), dtype=float)
                    penalty_jacobian[:, :d] = np.sqrt(mean_penalty_weight) * np.eye(d)
                    return np.vstack([data_jacobian, penalty_jacobian])

                res = least_squares(
                    residual,
                    theta_start,
                    jac=residual_jacobian,
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=ls_max_nfev,
                )

                m_new = res.x[:d]
                sigma_new = max(np.sqrt(np.exp(res.x[-1])), MIN_SIGMA)
                q_new = q_single(m_new, sigma_new)

                old_residual = R_minus_k - a_k * q_old
                new_residual = R_minus_k - a_k * q_new
                old_local = 0.5 * np.dot(old_residual, old_residual)
                new_local = 0.5 * np.dot(new_residual, new_residual)
                if mean_penalty_weight > 0:
                    old_delta = means[k] - means_reference[k]
                    new_delta = m_new - means_reference[k]
                    old_local += (
                        0.5 * mean_penalty_weight * np.dot(old_delta, old_delta)
                    )
                    new_local += (
                        0.5 * mean_penalty_weight * np.dot(new_delta, new_delta)
                    )

                if new_local <= old_local:
                    means[k] = m_new
                    sigmas[k] = sigma_new
                    Q[:, k] = q_new
                    # R_minus_k is the target with component k removed:
                    # R_minus_k = Z - pred_old + a_k * q_old.
                    # Restore the full prediction after replacing q_old by q_new.
                    pred = Z - R_minus_k + a_k * q_new

        amplitudes, _ = nnls(Q, Z)
        cur_data_obj = data_objective(Q, amplitudes)
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
        print(f"fit_dimension_free_one_s_gmm finished in {elapsed:.3f} seconds")

    return {
        "means": means,
        "sigmas": sigmas,
        "amplitudes": amplitudes,
        "objective": history[-1] if history else prev_obj,
        "history": np.asarray(history),
        "data_objective": (
            data_history[-1] if data_history else data_objective(Q, amplitudes)
        ),
        "data_history": np.asarray(data_history),
        "penalty_objective": (
            penalty_history[-1] if penalty_history else penalty_objective(means)
        ),
        "penalty_history": np.asarray(penalty_history),
        "n_iter": len(history),
        "elapsed_seconds": elapsed,
    }
