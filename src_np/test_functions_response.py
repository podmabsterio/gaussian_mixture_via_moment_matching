import numpy as np


def compute_average_kernel_count(
    data,
    test_centers,
    s,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
):
    """Return the average number of observations seen by a Gaussian kernel.

    The returned value is

        mean_j sum_i exp(-||data_i - test_centers_j||^2 / (2 s^2)).

    It is an effective (soft) neighbor count rather than a normalized kernel
    average.  In particular, when ``test_centers is data``, the diagonal
    contribution is one for every center.
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

        total += np.sum(center_sums)

    return total / J


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
                acc[~batch_base_mask] += base[~batch_base_mask] * lin[
                    ~batch_base_mask
                ]

        psi[j0:j1] = acc

    return psi
