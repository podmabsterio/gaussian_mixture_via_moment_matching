import numpy as np


def sample_gm_data(means, weights, covariances, num_samples, seed=1):
    rng = np.random.default_rng(seed)

    means = np.asarray(means, dtype=float)
    weights = np.asarray(weights, dtype=float)
    covariances = np.asarray(covariances, dtype=float)

    K, D = means.shape

    weights = weights / weights.sum()

    labels = rng.choice(K, size=num_samples, p=weights)

    X = np.empty((num_samples, D))

    for k in range(K):
        idx = labels == k
        n_k = idx.sum()

        if n_k > 0:
            X[idx] = rng.multivariate_normal(
                mean=means[k], cov=covariances[k], size=n_k
            )

    return X, labels
