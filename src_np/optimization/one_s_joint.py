import time

import numpy as np
from scipy.optimize import least_squares, nnls

from .utils import MIN_SIGMA


def fit_dimension_free_one_s_gmm_joint(
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
    """Fit one-bandwidth GMM moments with a joint geometry update.

    The amplitude block is solved exactly with NNLS.  Conditional on those
    amplitudes, all component means and log-variances are optimized together
    by one nonlinear least-squares problem.  The signature and result mapping
    intentionally match :func:`fit_dimension_free_one_s_gmm`.

    ``active_tol`` is accepted for interface compatibility.  No explicit
    active-component filtering is needed in the joint step: a zero amplitude
    naturally gives the corresponding geometry block a zero data Jacobian.
    """
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

    assert (
        X.ndim == 2
        and U.shape == X.shape
        and Z.shape == (X.shape[0],)
        and means.ndim == 2
        and means.shape[1] == X.shape[1]
        and sigmas.shape == (means.shape[0],)
        and np.isscalar(s)
        and s > 0
        and np.all(np.isfinite(sigmas))
        and np.all(sigmas > 0)
        and (amplitudes is None or amplitudes.shape == (means.shape[0],))
        and means_reference.shape == means.shape
        and mean_penalty_weight >= 0
        and active_tol >= 0
        and objective_rtol >= 0
        and objective_atol >= 0
        and convergence_patience >= 1
    )

    t0 = time.time()
    J, d = X.shape
    K = means.shape[0]
    s2 = float(s) ** 2
    sqrt_penalty_weight = np.sqrt(mean_penalty_weight)
    base_mask = np.linalg.norm(U, axis=1) <= np.finfo(float).tiny

    if amplitudes is None:
        amplitudes = np.ones(K) / K

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
        residual = Z - Q @ a
        return 0.5 * np.dot(residual, residual)

    def penalty_objective(means_):
        delta = means_ - means_reference
        return 0.5 * mean_penalty_weight * np.sum(delta * delta)

    def objective(Q, a, means_):
        return data_objective(Q, a) + penalty_objective(means_)

    def pack_geometry(means_, sigmas_):
        sigmas_ = np.maximum(sigmas_, MIN_SIGMA)
        eta = np.log(sigmas_ * sigmas_)
        return np.column_stack([means_, eta]).reshape(-1)

    def unpack_geometry(theta):
        component_params = theta.reshape(K, d + 1)
        means_ = component_params[:, :d]
        variances_ = np.maximum(np.exp(component_params[:, -1]), MIN_SIGMA * MIN_SIGMA)
        sigmas_ = np.sqrt(variances_)
        return means_, sigmas_, variances_

    penalty_jacobian = None
    if mean_penalty_weight > 0:
        penalty_jacobian = np.zeros((K * d, K * (d + 1)), dtype=float)
        component_indices = np.repeat(np.arange(K), d)
        coordinate_indices = np.tile(np.arange(d), K)
        rows = np.arange(K * d)
        columns = component_indices * (d + 1) + coordinate_indices
        penalty_jacobian[rows, columns] = sqrt_penalty_weight

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
            theta_start = pack_geometry(means, sigmas)
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
                "Q": None,
                "data_jacobian": None,
            }

            def evaluate_model_and_jacobian(theta):
                cached_theta = evaluation_cache["theta"]
                if cached_theta is not None and np.array_equal(theta, cached_theta):
                    return (
                        evaluation_cache["means"],
                        evaluation_cache["sigmas"],
                        evaluation_cache["Q"],
                        evaluation_cache["data_jacobian"],
                    )

                means_, sigmas_, variances_ = unpack_geometry(theta)
                r = means_[None, :, :] - X[:, None, :]
                t = s2 + variances_
                h = np.einsum("jkd,jd->jk", r, U)
                norm2 = np.einsum("jkd,jkd->jk", r, r)
                base = np.exp(-0.5 * norm2 / t[None, :])
                scale = base * (s2 / t)[None, :]
                Q_ = scale * h
                Q_[base_mask] = base[base_mask]

                jac_q = np.empty((J, K, d + 1), dtype=float)
                jac_q[:, :, :d] = scale[:, :, None] * (
                    U[:, None, :] - (h / t[None, :])[:, :, None] * r
                )
                jac_q[:, :, -1] = (
                    Q_
                    * variances_[None, :]
                    * (-1.0 / t[None, :] + 0.5 * norm2 / (t[None, :] * t[None, :]))
                )
                if np.any(base_mask):
                    jac_q[base_mask, :, :d] = (
                        base[base_mask, :, None]
                        * (-r[base_mask] / t[None, :, None])
                    )
                    jac_q[base_mask, :, -1] = (
                        base[base_mask]
                        * variances_[None, :]
                        * 0.5
                        * norm2[base_mask]
                        / (t[None, :] * t[None, :])
                    )
                data_jacobian = -(jac_q * amplitudes[None, :, None]).reshape(
                    J, K * (d + 1)
                )

                evaluation_cache["theta"] = theta.copy()
                evaluation_cache["means"] = means_
                evaluation_cache["sigmas"] = sigmas_
                evaluation_cache["Q"] = Q_
                evaluation_cache["data_jacobian"] = data_jacobian
                return means_, sigmas_, Q_, data_jacobian

            def residual(theta):
                means_, _, Q_, _ = evaluate_model_and_jacobian(theta)
                data_residual = Z - Q_ @ amplitudes
                if mean_penalty_weight == 0:
                    return data_residual
                penalty_residual = sqrt_penalty_weight * (
                    means_ - means_reference
                ).reshape(-1)
                return np.concatenate([data_residual, penalty_residual])

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

            means_new, sigmas_new, Q_new, _ = evaluate_model_and_jacobian(res.x)
            old_local = objective(Q, amplitudes, means)
            new_local = objective(Q_new, amplitudes, means_new)
            if new_local <= old_local:
                means = means_new.copy()
                sigmas = sigmas_new.copy()
                Q = Q_new.copy()

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
        print(
            "fit_dimension_free_one_s_gmm_joint finished " f"in {elapsed:.3f} seconds"
        )

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
