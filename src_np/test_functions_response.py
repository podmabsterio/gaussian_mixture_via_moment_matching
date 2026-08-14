import numpy as np


def _validate_leave_out_indices(leave_out_indices, n_tests, n_samples):
    if leave_out_indices is None:
        return None

    indices = np.asarray(leave_out_indices)
    if indices.shape != (n_tests,):
        raise ValueError("leave_out_indices must have one entry per test center")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("leave_out_indices must contain integer sample indices")
    indices = indices.astype(np.intp, copy=False)
    if np.any(indices < -1) or np.any(indices >= n_samples):
        raise ValueError("leave_out_indices entries must be -1 or valid sample indices")
    if n_samples < 2 and np.any(indices >= 0):
        raise ValueError("leave-one-out responses require at least two samples")
    return indices


def _self_kernel_values(data, centers, indices, s2):
    values = np.zeros(centers.shape[0], dtype=float)
    active = indices >= 0
    if np.any(active):
        differences = data[indices[active]] - centers[active]
        values[active] = np.exp(-0.5 * np.sum(differences * differences, axis=1) / s2)
    return values


def compute_Z_moment(
    data,
    test_centers,
    test_directions,
    s,
    moment_order,
    second_test_directions=None,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
    leave_out_indices=None,
):
    """Compute empirical Gaussian-kernel moments of order zero, one, or two.

    For a center ``c`` and directions ``u`` and ``v``, the supported test
    functions are

    ``K_s(x-c)``, ``<x-c,u> K_s(x-c)``, and
    ``<x-c,u><x-c,v> K_s(x-c)``.

    Passing ``v=u`` gives the squared-direction second moment.  A distinct
    ``v`` gives the two-direction moment from Lemma 2.1 in ``main.pdf``.

    ``leave_out_indices[j]`` may identify the observation from which center
    ``j`` was constructed.  Its contribution is then removed and that response
    is normalized by ``n - 1``.  Use ``-1`` for centers with no associated
    observation.
    """
    data = np.asarray(data, dtype=float)
    centers = np.asarray(test_centers, dtype=float)
    directions = np.asarray(test_directions, dtype=float)
    order = int(moment_order)
    s = float(s)

    if order not in (0, 1, 2):
        raise ValueError("moment_order must be 0, 1, or 2")
    if data.ndim != 2 or data.shape[0] == 0 or centers.ndim != 2:
        raise ValueError("data and test_centers must be two-dimensional")
    if centers.shape[0] == 0 or data.shape[1] != centers.shape[1]:
        raise ValueError("test_centers must be non-empty and match data dimension")
    if directions.shape != centers.shape:
        raise ValueError("test_directions must match test_centers")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    if data_batch_size < 1 or max_block_entries < 1:
        raise ValueError("batch sizes must be positive")

    if order == 2:
        if second_test_directions is None:
            second_directions = directions
        else:
            second_directions = np.asarray(second_test_directions, dtype=float)
        if second_directions.shape != centers.shape:
            raise ValueError("second_test_directions must match test_centers")
    else:
        second_directions = None

    n = data.shape[0]
    J = centers.shape[0]
    leave_out_indices = _validate_leave_out_indices(leave_out_indices, J, n)
    s2 = s * s
    if test_batch_size is None:
        test_batch_size = max(
            1,
            min(J, max_block_entries // max(1, min(n, data_batch_size))),
        )
    if test_batch_size < 1:
        raise ValueError("test_batch_size must be positive")

    result = np.zeros(J, dtype=float)
    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = centers[j0:j1]
        U = directions[j0:j1]
        V = None if second_directions is None else second_directions[j0:j1]
        c2 = np.sum(C * C, axis=1)
        cu = np.sum(C * U, axis=1)
        cv = None if V is None else np.sum(C * V, axis=1)
        accumulator = np.zeros(j1 - j0, dtype=float)

        for i0 in range(0, n, data_batch_size):
            i1 = min(n, i0 + data_batch_size)
            Y = data[i0:i1]
            y2 = np.sum(Y * Y, axis=1)
            kernel = y2[:, None] + c2[None, :] - 2.0 * (Y @ C.T)
            np.maximum(kernel, 0.0, out=kernel)
            kernel *= -0.5 / s2
            np.exp(kernel, out=kernel)

            if order == 0:
                accumulator += np.sum(kernel, axis=0)
                continue

            first_projection = Y @ U.T
            first_projection -= cu[None, :]
            if order == 1:
                accumulator += np.sum(first_projection * kernel, axis=0)
                continue

            second_projection = Y @ V.T
            second_projection -= cv[None, :]
            accumulator += np.sum(
                first_projection * second_projection * kernel,
                axis=0,
            )

        result[j0:j1] = accumulator / n

    if leave_out_indices is not None:
        active = leave_out_indices >= 0
        if np.any(active):
            self_kernel = _self_kernel_values(
                data,
                centers,
                leave_out_indices,
                s2,
            )
            if order == 0:
                self_contribution = self_kernel
            else:
                differences = data[leave_out_indices[active]] - centers[active]
                first_projection = np.einsum(
                    "jd,jd->j",
                    differences,
                    directions[active],
                )
                self_contribution = np.zeros(J, dtype=float)
                if order == 1:
                    self_contribution[active] = first_projection * self_kernel[active]
                else:
                    second_projection = np.einsum(
                        "jd,jd->j",
                        differences,
                        second_directions[active],
                    )
                    self_contribution[active] = (
                        first_projection * second_projection * self_kernel[active]
                    )
            result[active] = (n * result[active] - self_contribution[active]) / (n - 1)

    return result


def compute_Z_radial_second_moment(
    data,
    test_centers,
    s,
    compensated=False,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
    leave_out_indices=None,
):
    """Compute normalized radial second-moment responses.

    For every center ``c`` the test function is

    ``||x-c||**2 / (d * s**2) * K_s(x-c)``.

    With ``compensated=True``, ``K_s(x-c)`` is subtracted.  The resulting
    test function has zero integral over :math:`R^d`.  The normalization by
    both ``d`` and ``s**2`` is intrinsic to this family and is therefore
    applied independently of the model's directional-moment scaling option.
    """
    data = np.asarray(data, dtype=float)
    centers = np.asarray(test_centers, dtype=float)
    s = float(s)

    if data.ndim != 2 or data.shape[0] == 0 or centers.ndim != 2:
        raise ValueError("data and test_centers must be two-dimensional")
    if centers.shape[0] == 0 or data.shape[1] != centers.shape[1]:
        raise ValueError("test_centers must be non-empty and match data dimension")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    if data_batch_size < 1 or max_block_entries < 1:
        raise ValueError("batch sizes must be positive")

    n, d = data.shape
    J = centers.shape[0]
    leave_out_indices = _validate_leave_out_indices(leave_out_indices, J, n)
    s2 = s * s
    radial_scale = 1.0 / (d * s2)
    if test_batch_size is None:
        test_batch_size = max(
            1,
            min(J, max_block_entries // max(1, min(n, data_batch_size))),
        )
    if test_batch_size < 1:
        raise ValueError("test_batch_size must be positive")

    result = np.zeros(J, dtype=float)
    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = centers[j0:j1]
        c2 = np.sum(C * C, axis=1)
        accumulator = np.zeros(j1 - j0, dtype=float)

        for i0 in range(0, n, data_batch_size):
            i1 = min(n, i0 + data_batch_size)
            Y = data[i0:i1]
            y2 = np.sum(Y * Y, axis=1)
            squared_distance = y2[:, None] + c2[None, :] - 2.0 * (Y @ C.T)
            np.maximum(squared_distance, 0.0, out=squared_distance)
            kernel = np.exp(-0.5 * squared_distance / s2)
            radial_factor = radial_scale * squared_distance
            if compensated:
                radial_factor -= 1.0
            accumulator += np.sum(radial_factor * kernel, axis=0)

        result[j0:j1] = accumulator / n

    if leave_out_indices is not None:
        active = leave_out_indices >= 0
        if np.any(active):
            differences = data[leave_out_indices[active]] - centers[active]
            squared_distance = np.sum(differences * differences, axis=1)
            self_kernel = np.exp(-0.5 * squared_distance / s2)
            self_factor = radial_scale * squared_distance
            if compensated:
                self_factor -= 1.0
            self_contribution = self_factor * self_kernel
            result[active] = (n * result[active] - self_contribution) / (n - 1)

    return result


def compute_average_kernel_count(
    data,
    test_centers,
    s,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
    leave_out_indices=None,
):
    """Return the average number of observations seen by a Gaussian kernel.

    The returned value is

        mean_j sum_i exp(-||data_i - test_centers_j||^2 / (2 s^2)).

    It is an effective (soft) neighbor count rather than a normalized kernel
    average.  In particular, when ``test_centers is data``, the diagonal
    contribution is one for every center unless its index is supplied through
    ``leave_out_indices``.  Use ``-1`` for independent centers.
    """
    data = np.asarray(data, dtype=float)
    X = np.asarray(test_centers, dtype=float)
    s = float(s)

    assert (
        data.ndim == 2
        and X.ndim == 2
        and data.shape[1] == X.shape[1]
        and X.shape[0] > 0
        and s > 0
        and data_batch_size > 0
        and max_block_entries > 0
    )

    n = data.shape[0]
    J = X.shape[0]
    leave_out_indices = _validate_leave_out_indices(leave_out_indices, J, n)
    s2 = s * s

    if test_batch_size is None:
        test_batch_size = max(1, min(J, max_block_entries // min(n, data_batch_size)))

    total = 0.0
    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = X[j0:j1]
        c2 = np.sum(C * C, axis=1)

        center_sums = np.zeros(j1 - j0, dtype=float)
        for i0 in range(0, n, data_batch_size):
            i1 = min(n, i0 + data_batch_size)
            Y = data[i0:i1]
            y2 = np.sum(Y * Y, axis=1)
            dist2 = y2[:, None] + c2[None, :] - 2.0 * (Y @ C.T)
            np.maximum(dist2, 0.0, out=dist2)
            dist2 *= -0.5 / s2
            np.exp(dist2, out=dist2)
            center_sums += np.sum(dist2, axis=0)

        if leave_out_indices is not None:
            batch_indices = leave_out_indices[j0:j1]
            center_sums -= _self_kernel_values(data, C, batch_indices, s2)

        total += np.sum(center_sums)

    return total / J


def compute_kernel_counts(
    data,
    test_centers,
    s,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
    leave_out_indices=None,
):
    """Return the soft number of observations visible from every center.

    For every test center ``c_j``, the returned score is

        sum_i exp(-||data_i - c_j||^2 / (2 s^2)).

    Unlike :func:`compute_average_kernel_count`, this function preserves the
    individual score of every center.  Computation is batched and does not
    materialize the full ``n_samples x n_centers`` kernel matrix.  Associated
    observations supplied through ``leave_out_indices`` are excluded.
    """
    data = np.asarray(data, dtype=float)
    centers = np.asarray(test_centers, dtype=float)
    s = float(s)

    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must be a non-empty two-dimensional array")
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != data.shape[1]:
        raise ValueError("test_centers must be non-empty and match the data dimension")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(centers)):
        raise ValueError("data and test_centers must contain only finite values")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("s must be finite and positive")
    if data_batch_size < 1 or max_block_entries < 1:
        raise ValueError("batch sizes must be positive")

    n = data.shape[0]
    J = centers.shape[0]
    leave_out_indices = _validate_leave_out_indices(leave_out_indices, J, n)
    s2 = s * s
    if test_batch_size is None:
        test_batch_size = max(
            1,
            min(J, max_block_entries // max(1, min(n, data_batch_size))),
        )
    if test_batch_size < 1:
        raise ValueError("test_batch_size must be positive")

    counts = np.empty(J, dtype=float)
    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = centers[j0:j1]
        c2 = np.sum(C * C, axis=1)
        center_counts = np.zeros(j1 - j0, dtype=float)

        for i0 in range(0, n, data_batch_size):
            i1 = min(n, i0 + data_batch_size)
            Y = data[i0:i1]
            y2 = np.sum(Y * Y, axis=1)
            dist2 = y2[:, None] + c2[None, :] - 2.0 * (Y @ C.T)
            np.maximum(dist2, 0.0, out=dist2)
            dist2 *= -0.5 / s2
            np.exp(dist2, out=dist2)
            center_counts += np.sum(dist2, axis=0)

        if leave_out_indices is not None:
            batch_indices = leave_out_indices[j0:j1]
            center_counts -= _self_kernel_values(
                data,
                C,
                batch_indices,
                s2,
            )

        counts[j0:j1] = center_counts

    return counts


def compute_Z_one_s(
    data,
    test_centers,
    test_directions,
    s,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
):
    data = np.asarray(data, dtype=float)
    X = np.asarray(test_centers, dtype=float)
    U = np.asarray(test_directions, dtype=float)
    s = float(s)

    assert (
        data.ndim == 2
        and X.ndim == 2
        and U.shape == X.shape
        and data.shape[1] == X.shape[1]
        and s > 0
        and data_batch_size > 0
        and max_block_entries > 0
    )

    n, d = data.shape
    J = X.shape[0]
    s2 = s * s

    if test_batch_size is None:
        test_batch_size = max(1, min(J, max_block_entries // min(n, data_batch_size)))

    direction_norms = np.linalg.norm(U, axis=1)
    base_mask = direction_norms <= np.finfo(float).tiny

    Z = np.zeros(J, dtype=float)

    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = X[j0:j1]
        V = U[j0:j1]
        batch_base_mask = base_mask[j0:j1]

        c2 = np.sum(C * C, axis=1)
        cu = np.sum(C * V, axis=1)

        acc = np.zeros(j1 - j0, dtype=float)

        for i0 in range(0, n, data_batch_size):
            i1 = min(n, i0 + data_batch_size)
            Y = data[i0:i1]

            y2 = np.sum(Y * Y, axis=1)

            dist2 = y2[:, None] + c2[None, :] - 2.0 * (Y @ C.T)
            np.maximum(dist2, 0.0, out=dist2)

            lin = Y @ V.T
            lin -= cu[None, :]

            dist2 *= -0.5 / s2
            np.exp(dist2, out=dist2)

            if np.any(batch_base_mask):
                acc[batch_base_mask] += np.sum(dist2[:, batch_base_mask], axis=0)

            if np.any(~batch_base_mask):
                lin *= dist2
                acc[~batch_base_mask] += np.sum(lin[:, ~batch_base_mask], axis=0)

        Z[j0:j1] = acc / n

    return Z


def compute_psi_one_s_isotropic_gmm(
    test_centers,
    test_directions,
    s,
    means,
    sigmas,
    weights=None,
    amplitudes=None,
    test_batch_size=8192,
):
    X = np.asarray(test_centers, dtype=float)
    U = np.asarray(test_directions, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    s = float(s)

    assert (
        X.ndim == 2
        and U.shape == X.shape
        and means.ndim == 2
        and means.shape[1] == X.shape[1]
        and sigmas.shape == (means.shape[0],)
        and np.all(sigmas > 0)
        and s > 0
        and test_batch_size > 0
        and ((weights is None) ^ (amplitudes is None))
    )

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        assert weights.shape == sigmas.shape
    else:
        amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
        assert amplitudes.shape == sigmas.shape

    J, d = X.shape
    K = means.shape[0]
    s2 = s * s
    v = sigmas * sigmas
    t = s2 + v

    if weights is not None:
        coef = weights * np.exp(-0.5 * d * np.log1p(v / s2)) * (s2 / t)
        base_coef = weights * np.exp(-0.5 * d * np.log1p(v / s2))
    else:
        coef = amplitudes * (s2 / t)
        base_coef = amplitudes

    m2 = np.sum(means * means, axis=1)
    psi = np.zeros(J, dtype=float)
    base_mask = np.linalg.norm(U, axis=1) <= np.finfo(float).tiny

    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = X[j0:j1]
        V = U[j0:j1]
        batch_base_mask = base_mask[j0:j1]

        c2 = np.sum(C * C, axis=1)
        cu = np.sum(C * V, axis=1)

        dist2 = c2[:, None] + m2[None, :] - 2.0 * (C @ means.T)
        np.maximum(dist2, 0.0, out=dist2)

        lin = V @ means.T
        lin -= cu[:, None]

        dist2 *= -0.5 / t[None, :]
        np.exp(dist2, out=dist2)

        psi_batch = psi[j0:j1]

        if np.any(batch_base_mask):
            base_values = dist2[batch_base_mask].copy()
            base_values *= base_coef[None, :]
            psi_batch[batch_base_mask] = np.sum(base_values, axis=1)

        if np.any(~batch_base_mask):
            dir_values = dist2[~batch_base_mask].copy()
            dir_values *= lin[~batch_base_mask]
            dir_values *= coef[None, :]
            psi_batch[~batch_base_mask] = np.sum(dir_values, axis=1)

    return psi


def compute_psi_one_s_full_covariance_gmm(
    test_centers,
    test_directions,
    s,
    means,
    covariances,
    weights=None,
    amplitudes=None,
    test_batch_size=8192,
):
    X = np.asarray(test_centers, dtype=float)
    U = np.asarray(test_directions, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    s = float(s)

    assert (
        X.ndim == 2
        and U.shape == X.shape
        and means.ndim == 2
        and means.shape[1] == X.shape[1]
        and covariances.shape == (means.shape[0], means.shape[1], means.shape[1])
        and s > 0
        and test_batch_size > 0
        and ((weights is None) ^ (amplitudes is None))
    )

    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        assert weights.shape == (means.shape[0],)
    else:
        amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
        assert amplitudes.shape == (means.shape[0],)

    J, d = X.shape
    K = means.shape[0]
    s2 = s * s

    eye = np.eye(d)
    B = np.empty_like(covariances)
    det_coef = np.empty(K, dtype=float)

    for k in range(K):
        scaled = eye + covariances[k] / s2
        sign, logdet = np.linalg.slogdet(scaled)
        assert sign > 0
        B[k] = np.linalg.solve(scaled, eye)
        det_coef[k] = np.exp(-0.5 * logdet)

    if weights is not None:
        coef = weights * det_coef
    else:
        coef = amplitudes

    psi = np.zeros(J, dtype=float)
    base_mask = np.linalg.norm(U, axis=1) <= np.finfo(float).tiny

    for j0 in range(0, J, test_batch_size):
        j1 = min(J, j0 + test_batch_size)
        C = X[j0:j1]
        V = U[j0:j1]
        batch_base_mask = base_mask[j0:j1]

        acc = np.zeros(j1 - j0, dtype=float)
        for k in range(K):
            diff = means[k] - C
            bdiff = diff @ B[k]
            quad = np.sum(diff * bdiff, axis=1)
            lin = np.sum(bdiff * V, axis=1)
            base = coef[k] * np.exp(-0.5 * quad / s2)
            if np.any(batch_base_mask):
                acc[batch_base_mask] += base[batch_base_mask]
            if np.any(~batch_base_mask):
                acc[~batch_base_mask] += base[~batch_base_mask] * lin[~batch_base_mask]

        psi[j0:j1] = acc

    return psi
