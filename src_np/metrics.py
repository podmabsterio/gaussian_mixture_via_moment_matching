import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment


def gmm_test_log_likelihood(
    pred_means,
    pred_sigmas,
    pred_weights,
    true_means,
    true_sigmas,
    true_weights,
    n_test=10000,
    batch_size=8192,
    random_state=None,
):
    pred_means = np.asarray(pred_means, dtype=float)
    pred_sigmas = np.asarray(pred_sigmas, dtype=float).reshape(-1)
    pred_weights = np.asarray(pred_weights, dtype=float).reshape(-1)

    true_means = np.asarray(true_means, dtype=float)
    true_sigmas = np.asarray(true_sigmas, dtype=float).reshape(-1)
    true_weights = np.asarray(true_weights, dtype=float).reshape(-1)

    assert (
        pred_means.ndim == 2
        and true_means.ndim == 2
        and pred_means.shape[1] == true_means.shape[1]
        and pred_sigmas.shape == (pred_means.shape[0],)
        and pred_weights.shape == (pred_means.shape[0],)
        and true_sigmas.shape == (true_means.shape[0],)
        and true_weights.shape == (true_means.shape[0],)
        and np.all(pred_sigmas > 0)
        and np.all(true_sigmas > 0)
        and np.all(pred_weights >= 0)
        and np.all(true_weights >= 0)
        and np.sum(pred_weights) > 0
        and np.sum(true_weights) > 0
        and n_test > 0
        and batch_size > 0
    )

    rng = np.random.default_rng(random_state)

    pred_weights = pred_weights / np.sum(pred_weights)
    true_weights = true_weights / np.sum(true_weights)

    n_test = int(n_test)
    d = true_means.shape[1]
    K_true = true_means.shape[0]
    K_pred = pred_means.shape[0]

    labels = rng.choice(K_true, size=n_test, p=true_weights)
    test_data = true_means[labels] + true_sigmas[labels, None] * rng.normal(
        size=(n_test, d)
    )

    log_weights = np.log(np.maximum(pred_weights, 1e-300))
    log_norm = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(pred_sigmas)
    m2 = np.sum(pred_means * pred_means, axis=1)

    total = 0.0

    for i0 in range(0, n_test, batch_size):
        i1 = min(n_test, i0 + batch_size)
        Y = test_data[i0:i1]

        y2 = np.sum(Y * Y, axis=1)
        dist2 = y2[:, None] + m2[None, :] - 2.0 * (Y @ pred_means.T)
        np.maximum(dist2, 0.0, out=dist2)

        log_comp = (
            log_weights[None, :]
            + log_norm[None, :]
            - 0.5 * dist2 / (pred_sigmas * pred_sigmas)[None, :]
        )

        total += np.sum(logsumexp(log_comp, axis=1))

    return total / n_test


def matched_center_distance(
    pred_means,
    true_means,
    return_assignment=False,
):
    pred_means = np.asarray(pred_means, dtype=float)
    true_means = np.asarray(true_means, dtype=float)

    assert (
        pred_means.ndim == 2
        and true_means.ndim == 2
        and pred_means.shape == true_means.shape
    )

    diff = pred_means[:, None, :] - true_means[None, :, :]
    cost = np.linalg.norm(diff, axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    distance = np.mean(cost[row_ind, col_ind])

    if return_assignment:
        return distance, col_ind[np.argsort(row_ind)]

    return distance
