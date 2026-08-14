import numpy as np

from experiments.synthetic_data._utils import make_rng


def sample_student_t_mixture(
    means,
    covariances,
    labels,
    degrees_of_freedom,
    seed=None,
):
    """Sample an elliptical Student-t mixture with prescribed covariances.

    For ``degrees_of_freedom > 2``, multiplying a Gaussian residual by
    ``sqrt((df - 2) / chi2(df))`` makes its covariance equal to the supplied
    covariance matrix. Thus the returned distribution is heavy-tailed while
    retaining the same component means and second moments as the Gaussian
    target.
    """
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    df = float(degrees_of_freedom)

    if not np.isfinite(df) or df <= 2:
        raise ValueError("degrees_of_freedom must be finite and greater than 2")
    if means.ndim != 2:
        raise ValueError("means must be two-dimensional")
    n_components, n_features = means.shape
    if covariances.shape != (n_components, n_features, n_features):
        raise ValueError("covariances must match means")
    if np.any(labels < 0) or np.any(labels >= n_components):
        raise ValueError("labels must contain valid component indices")

    rng = make_rng(seed, salt=101)
    X = np.empty((labels.size, n_features), dtype=float)
    for component in range(n_components):
        mask = labels == component
        count = int(np.sum(mask))
        if count == 0:
            continue
        gaussian = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=covariances[component],
            size=count,
        )
        scales = np.sqrt((df - 2.0) / rng.chisquare(df, size=count))
        X[mask] = means[component] + gaussian * scales[:, None]
    return X
