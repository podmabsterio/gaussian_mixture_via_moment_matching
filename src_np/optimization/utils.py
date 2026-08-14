import numpy as np


MIN_SIGMA = 1e-6
RADIAL_SECOND_MOMENT = "radial_second"


def _evaluate_moment_Q_and_jacobian(
    test_centers,
    test_directions,
    second_test_directions,
    means,
    variances,
    s,
    moment_order,
    *,
    compute_jacobian,
    moment_family=None,
    compensated=False,
):
    """Evaluate a dimension-free moment block and its geometry Jacobian.

    The columns of the returned matrix correspond to mixture components.  An
    amplitude column has the common dimension-free factor

    ``weight * (1 + variance / s**2)**(-d / 2)``

    removed from it.  Consequently, zeroth-, first-, and second-order blocks
    for the same bandwidth can share one amplitude vector.

    The final Jacobian coordinate is with respect to ``log(variance)`` rather
    than ``sigma``.  For a directional second-order block,
    ``second_test_directions`` may equal ``test_directions`` (squared
    directional moment) or contain a second set of directions
    (cross-directional moment).

    ``moment_family='radial_second'`` evaluates the intrinsically normalized
    radial test function ``||x-c||**2 / (d * s**2) * K_s(x-c)``.  If
    ``compensated=True``, its zeroth-order kernel part is subtracted so that
    the test function integrates to zero over :math:`R^d`.
    """
    X = np.asarray(test_centers, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float).reshape(-1)
    order = int(moment_order)
    family = order if moment_family is None else moment_family
    is_radial = family == RADIAL_SECOND_MOMENT

    if order not in (0, 1, 2):
        raise ValueError("moment_order must be 0, 1, or 2")
    if is_radial:
        if order != 2:
            raise ValueError("radial_second moment blocks must have moment_order=2")
        if compensated not in (False, True):
            raise ValueError("compensated must be a boolean")
        U = None
    else:
        if family != order:
            raise ValueError("unsupported moment_family")
        if compensated:
            raise ValueError("compensation is only defined for radial moments")
        U = np.asarray(test_directions, dtype=float)
    if X.ndim != 2 or (U is not None and U.shape != X.shape):
        raise ValueError("test centers and directions must have matching 2D shapes")
    if means.ndim != 2 or means.shape[1] != X.shape[1]:
        raise ValueError("means must match the test-center dimension")
    if variances.shape != (means.shape[0],) or np.any(variances <= 0):
        raise ValueError("variances must be positive and match the means")

    if order == 2 and not is_radial:
        V = np.asarray(second_test_directions, dtype=float)
        if V.shape != X.shape:
            raise ValueError(
                "second_test_directions must match test_centers for order 2"
            )
    else:
        V = None

    s2 = float(s) ** 2
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("s must be finite and positive")

    J, d = X.shape
    if d == 0:
        raise ValueError("moment dimension must be positive")
    K = means.shape[0]
    r = means[None, :, :] - X[:, None, :]
    norm2 = np.einsum("jkd,jkd->jk", r, r)
    t = s2 + variances
    rho = s2 / t
    base = np.exp(-0.5 * norm2 / t[None, :])

    if is_radial:
        radial_quadratic_scale = s2 / (d * t * t)
        factor = (
            radial_quadratic_scale[None, :] * norm2 + variances[None, :] / t[None, :]
        )
        if compensated:
            factor -= 1.0
        Q = base * factor
    elif order == 0:
        factor = np.ones((J, K), dtype=float)
        Q = base
    else:
        h = np.einsum("jkd,jd->jk", r, U)
        if order == 1:
            factor = rho[None, :] * h
        else:
            g = np.einsum("jkd,jd->jk", r, V)
            direction_inner_product = np.einsum("jd,jd->j", U, V)
            factor = (
                rho[None, :] ** 2 * h * g
                + variances[None, :] * rho[None, :] * direction_inner_product[:, None]
            )
        Q = base * factor

    if not compute_jacobian:
        return Q, None

    jac_q = np.empty((J, K, d + 1), dtype=float)
    if is_radial:
        factor_mean_derivative = 2.0 * radial_quadratic_scale[None, :, None] * r
        factor_eta_derivative = variances[None, :] * (
            -2.0 * radial_quadratic_scale[None, :] * norm2 / t[None, :]
            + s2 / (t[None, :] * t[None, :])
        )
    elif order == 0:
        factor_mean_derivative = np.zeros((J, K, d), dtype=float)
        factor_eta_derivative = np.zeros((J, K), dtype=float)
    elif order == 1:
        factor_mean_derivative = np.broadcast_to(
            rho[None, :, None] * U[:, None, :],
            (J, K, d),
        )
        factor_eta_derivative = -factor * variances[None, :] / t[None, :]
    else:
        first_second_product = rho[None, :] ** 2 * h * g
        factor_mean_derivative = rho[None, :, None] ** 2 * (
            U[:, None, :] * g[:, :, None] + V[:, None, :] * h[:, :, None]
        )
        factor_eta_derivative = (
            -2.0 * first_second_product * variances[None, :] / t[None, :]
            + variances[None, :] * rho[None, :] ** 2 * direction_inner_product[:, None]
        )

    jac_q[:, :, :d] = base[:, :, None] * (
        factor_mean_derivative - factor[:, :, None] * r / t[None, :, None]
    )
    jac_q[:, :, -1] = base * (
        factor_eta_derivative
        + factor * variances[None, :] * 0.5 * norm2 / (t[None, :] * t[None, :])
    )
    return Q, jac_q


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
        jac_q[base_mask, :, :d] = base[base_mask, :, None] * (
            -r[base_mask] / t[None, :, None]
        )
        jac_q[base_mask, :, -1] = (
            base[base_mask]
            * variances[None, :]
            * 0.5
            * norm2[base_mask]
            / (t[None, :] * t[None, :])
        )

    return Q, jac_q
