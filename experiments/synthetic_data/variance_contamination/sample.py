import numpy as np

from experiments.synthetic_data._utils import contamination_mask, make_rng


def contaminate_component_variances(
    X,
    labels,
    means,
    covariances,
    contamination_fraction,
    variance_inflation,
    seed=None,
):
    """Replace selected observations by broad same-component Gaussians."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    inflation = float(variance_inflation)

    if X.ndim != 2 or labels.shape != (X.shape[0],):
        raise ValueError("X and labels have incompatible shapes")
    if means.ndim != 2 or means.shape[1] != X.shape[1]:
        raise ValueError("means must match X")
    if covariances.shape != (means.shape[0], X.shape[1], X.shape[1]):
        raise ValueError("covariances must match means")
    if not np.isfinite(inflation) or inflation <= 1:
        raise ValueError("variance_inflation must be finite and greater than 1")

    rng = make_rng(seed, salt=201)
    mask = contamination_mask(X.shape[0], contamination_fraction, rng)
    contaminated = X.copy()
    for component in range(means.shape[0]):
        component_mask = mask & (labels == component)
        count = int(np.sum(component_mask))
        if count:
            contaminated[component_mask] = rng.multivariate_normal(
                mean=means[component],
                cov=inflation * covariances[component],
                size=count,
            )
    return contaminated, mask
