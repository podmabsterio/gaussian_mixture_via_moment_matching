import numpy as np

from experiments.synthetic_data._utils import contamination_mask, make_rng


def contaminate_with_radial_outliers(
    X,
    means,
    weights,
    covariances,
    contamination_fraction,
    min_radius_scale,
    max_radius_scale,
    seed=None,
):
    """Replace observations by points in a remote spherical shell."""
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    covariances = np.asarray(covariances, dtype=float)
    min_scale = float(min_radius_scale)
    max_scale = float(max_radius_scale)

    if X.ndim != 2 or means.ndim != 2 or means.shape[1] != X.shape[1]:
        raise ValueError("X and means have incompatible shapes")
    if weights.shape != (means.shape[0],):
        raise ValueError("weights must match means")
    if covariances.shape != (means.shape[0], X.shape[1], X.shape[1]):
        raise ValueError("covariances must match means")
    if (
        not np.isfinite(min_scale)
        or not np.isfinite(max_scale)
        or min_scale <= 0
        or max_scale <= min_scale
    ):
        raise ValueError(
            "radius scales must be finite and satisfy 0 < min < max"
        )

    rng = make_rng(seed, salt=401)
    mask = contamination_mask(X.shape[0], contamination_fraction, rng)
    count = int(np.sum(mask))

    mixture_center = weights @ means
    component_radius = float(
        np.max(np.linalg.norm(means - mixture_center[None, :], axis=1))
    )
    component_scale = float(
        np.sqrt(
            np.max(np.trace(covariances, axis1=1, axis2=2)) / X.shape[1]
        )
    )
    min_radius = component_radius + min_scale * component_scale
    max_radius = component_radius + max_scale * component_scale

    directions = rng.normal(size=(count, X.shape[1]))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, np.finfo(float).tiny)
    radii = rng.uniform(min_radius, max_radius, size=count)

    contaminated = X.copy()
    contaminated[mask] = mixture_center + directions * radii[:, None]
    return contaminated, mask, mixture_center, min_radius, max_radius
