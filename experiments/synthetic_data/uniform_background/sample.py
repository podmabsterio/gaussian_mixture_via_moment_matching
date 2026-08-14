import numpy as np

from experiments.synthetic_data._utils import contamination_mask, make_rng


def contaminate_with_uniform_background(
    X,
    means,
    covariances,
    contamination_fraction,
    box_padding,
    seed=None,
):
    """Replace observations by diffuse uniform background points."""
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    padding = float(box_padding)

    if X.ndim != 2 or means.ndim != 2 or means.shape[1] != X.shape[1]:
        raise ValueError("X and means have incompatible shapes")
    if covariances.shape != (means.shape[0], X.shape[1], X.shape[1]):
        raise ValueError("covariances must match means")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("box_padding must be finite and positive")

    rng = make_rng(seed, salt=301)
    mask = contamination_mask(X.shape[0], contamination_fraction, rng)
    feature_scale = np.sqrt(
        np.maximum(
            np.max(np.diagonal(covariances, axis1=1, axis2=2), axis=0),
            np.finfo(float).tiny,
        )
    )
    lower = np.min(means, axis=0) - padding * feature_scale
    upper = np.max(means, axis=0) + padding * feature_scale

    contaminated = X.copy()
    contaminated[mask] = rng.uniform(
        lower,
        upper,
        size=(int(np.sum(mask)), X.shape[1]),
    )
    return contaminated, mask, lower, upper
