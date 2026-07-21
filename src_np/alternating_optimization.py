import time
import numpy as np
from scipy.optimize import nnls, least_squares


MIN_SIGMA = 1e-6


def _unpack_geometry(theta, n_components, n_features):
    component_params = theta.reshape(n_components, n_features + 1)
    means = component_params[:, :n_features]
    variances = np.maximum(
        np.exp(component_params[:, -1]),
        MIN_SIGMA * MIN_SIGMA,
    )
    sigmas = np.sqrt(variances)
    return means, sigmas, variances


def _pack_geometry(means, sigmas):
    sigmas = np.maximum(sigmas, MIN_SIGMA)
    eta = np.log(sigmas * sigmas)
    return np.column_stack([means, eta]).reshape(-1)


def _build_mean_penalty_jacobian(n_components, n_features, sqrt_penalty_weight):
    penalty_jacobian = np.zeros(
        (n_components * n_features, n_components * (n_features + 1)),
        dtype=float,
    )
    component_indices = np.repeat(np.arange(n_components), n_features)
    coordinate_indices = np.tile(np.arange(n_features), n_components)
    rows = np.arange(n_components * n_features)
    columns = component_indices * (n_features + 1) + coordinate_indices
    penalty_jacobian[rows, columns] = sqrt_penalty_weight
    return penalty_jacobian


def _evaluate_dimension_free_Q_and_jacobian(
    test_centers,
    test_directions,
    means,
    variances,
    s,
    *,
    compute_jacobian,
):
    X = test_centers
    U = test_directions
    s2 = float(s) ** 2
    base_mask = np.linalg.norm(U, axis=1) <= np.finfo(float).tiny

    J, d = X.shape
    K = means.shape[0]
    r = means[None, :, :] - X[:, None, :]
    t = s2 + variances
    h = np.einsum("jkd,jd->jk", r, U)
    norm2 = np.einsum("jkd,jkd->jk", r, r)
    base = np.exp(-0.5 * norm2 / t[None, :])
    scale = base * (s2 / t)[None, :]
    Q = scale * h
    Q[base_mask] = base[base_mask]

    if not compute_jacobian:
        return Q, None

    jac_q = np.empty((J, K, d + 1), dtype=float)
    jac_q[:, :, :d] = scale[:, :, None] * (
        U[:, None, :] - (h / t[None, :])[:, :, None] * r
    )
    jac_q[:, :, -1] = (
        Q
        * variances[None, :]
        * (-1.0 / t[None, :] + 0.5 * norm2 / (t[None, :] * t[None, :]))
    )
    if np.any(base_mask):
        jac_q[base_mask, :, :d] = (
            base[base_mask, :, None] * (-r[base_mask] / t[None, :, None])
        )
        jac_q[base_mask, :, -1] = (
            base[base_mask]
            * variances[None, :]
            * 0.5
            * norm2[base_mask]
            / (t[None, :] * t[None, :])
        )

    return Q, jac_q


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
