import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-10


def logsumexp(a, axis=0):
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(
        a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)), axis=axis
    )


def log_gaussian_pdf(X, mean, covariance):
    d = mean.shape[0]
    L = np.linalg.cholesky(covariance + EPS * np.eye(d))

    diff = X - mean
    solved = np.linalg.solve(L, diff.T)

    mahalanobis = np.sum(solved**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + mahalanobis)


def log_gm_pdf(X, means, weights, covariances):
    log_components = []

    for k in range(len(weights)):
        log_components.append(
            np.log(weights[k] + EPS) + log_gaussian_pdf(X, means[k], covariances[k])
        )

    return logsumexp(np.vstack(log_components), axis=0)


def sort_estimated_params_by_means(true_means, means, weights, covariances):
    cost = np.linalg.norm(true_means[:, None, :] - means[None, :, :], axis=2)

    true_indices, estimated_indices = linear_sum_assignment(cost)

    permutation = np.empty(len(true_means), dtype=int)
    permutation[true_indices] = estimated_indices

    return means[permutation], weights[permutation], covariances[permutation]


def log_component_densities(X, means, weights, covariances):
    log_components = []

    for k in range(len(weights)):
        log_components.append(
            np.log(weights[k] + EPS) + log_gaussian_pdf(X, means[k], covariances[k])
        )

    return np.vstack(log_components).T


def predict_gm_labels(X, means, weights, covariances):
    log_components = log_component_densities(X, means, weights, covariances)
    return np.argmax(log_components, axis=1)
