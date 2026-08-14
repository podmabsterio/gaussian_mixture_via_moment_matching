"""Sampling and density evaluation for directionally skewed mixtures."""

import numpy as np
from scipy.special import gammaln, logsumexp

from experiments.synthetic_data._utils import make_rng


def _validate_parameters(
    means,
    weights,
    covariances,
    latent_directions,
    gamma_shape,
):
    means = np.asarray(means, dtype=float)
    weights = np.asarray(weights, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    latent_directions = np.asarray(latent_directions, dtype=float)
    gamma_shape = float(gamma_shape)

    if means.ndim != 2:
        raise ValueError("means must be two-dimensional")
    n_components, n_features = means.shape
    if weights.shape != (n_components,):
        raise ValueError("weights must have one entry per component")
    if covariances.shape != (n_components, n_features, n_features):
        raise ValueError("covariances must match means")
    if latent_directions.shape != means.shape:
        raise ValueError("latent_directions must match means")
    if not np.all(np.isfinite(means)):
        raise ValueError("means must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    if not np.all(np.isfinite(covariances)):
        raise ValueError("covariances must be finite")
    if not np.all(np.isfinite(latent_directions)):
        raise ValueError("latent_directions must be finite")
    if not np.isfinite(gamma_shape) or gamma_shape <= 0:
        raise ValueError("gamma_shape must be finite and positive")

    direction_norms = np.linalg.norm(latent_directions, axis=1)
    if not np.allclose(direction_norms, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("latent_directions must be unit vectors")

    weights = weights / np.sum(weights)
    return (
        means,
        weights,
        covariances,
        latent_directions,
        gamma_shape,
    )


def sample_directional_gamma_mixture(
    means,
    weights,
    covariances,
    latent_directions,
    gamma_shape,
    num_samples=None,
    *,
    labels=None,
    seed=None,
    salt=501,
):
    """Sample a mixture with exact prescribed first and second moments.

    In whitened coordinates for component ``k`` the residual is

    ``u_k * S + (I - u_k u_k.T) * Z``,

    where ``Z`` is standard Gaussian and
    ``S = (Gamma(gamma_shape, 1) - gamma_shape) / sqrt(gamma_shape)``.
    Both terms are centered and their covariances add to the identity, while
    all non-Gaussian skewness is confined to ``u_k``.
    """
    (
        means,
        weights,
        covariances,
        latent_directions,
        gamma_shape,
    ) = _validate_parameters(
        means,
        weights,
        covariances,
        latent_directions,
        gamma_shape,
    )
    rng = make_rng(seed, salt=salt)

    if labels is None:
        if num_samples is None:
            raise ValueError("num_samples is required when labels are not supplied")
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        labels = rng.choice(len(weights), size=num_samples, p=weights)
    else:
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if num_samples is not None and int(num_samples) != labels.size:
            raise ValueError("num_samples must match the supplied labels")
        if np.any(labels < 0) or np.any(labels >= len(weights)):
            raise ValueError("labels must contain valid component indices")

    n_features = means.shape[1]
    X = np.empty((labels.size, n_features), dtype=float)
    sqrt_shape = np.sqrt(gamma_shape)

    for component in range(len(weights)):
        mask = labels == component
        count = int(np.sum(mask))
        if count == 0:
            continue

        direction = latent_directions[component]
        gaussian = rng.normal(size=(count, n_features))
        gaussian_orthogonal = gaussian - (
            gaussian @ direction
        )[:, None] * direction[None, :]
        skew_coordinate = (
            rng.gamma(shape=gamma_shape, scale=1.0, size=count) - gamma_shape
        ) / sqrt_shape
        whitened = (
            gaussian_orthogonal
            + skew_coordinate[:, None] * direction[None, :]
        )

        cholesky = np.linalg.cholesky(covariances[component])
        X[mask] = means[component] + whitened @ cholesky.T

    return X, labels


def log_directional_gamma_component_pdf(
    X,
    mean,
    covariance,
    latent_direction,
    gamma_shape,
):
    """Evaluate one directional-Gamma component density."""
    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    latent_direction = np.asarray(latent_direction, dtype=float)
    gamma_shape = float(gamma_shape)

    if X.ndim != 2 or X.shape[1] != mean.size:
        raise ValueError("X and mean have incompatible shapes")

    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, (X - mean).T).T
    skew_coordinate = whitened @ latent_direction
    orthogonal_squared_norm = np.maximum(
        np.sum(whitened * whitened, axis=1) - skew_coordinate**2,
        0.0,
    )

    sqrt_shape = np.sqrt(gamma_shape)
    gamma_value = gamma_shape + sqrt_shape * skew_coordinate
    log_skew_density = np.full(X.shape[0], -np.inf, dtype=float)
    inside_support = gamma_value > 0.0
    log_skew_density[inside_support] = (
        (gamma_shape - 1.0) * np.log(gamma_value[inside_support])
        - gamma_value[inside_support]
        - gammaln(gamma_shape)
        + np.log(sqrt_shape)
    )

    orthogonal_dimension = mean.size - 1
    log_orthogonal_density = -0.5 * (
        orthogonal_dimension * np.log(2.0 * np.pi)
        + orthogonal_squared_norm
    )
    log_abs_determinant = np.sum(np.log(np.diag(cholesky)))
    return log_skew_density + log_orthogonal_density - log_abs_determinant


def log_directional_gamma_mixture_pdf(
    X,
    means,
    weights,
    covariances,
    latent_directions,
    gamma_shape,
):
    """Evaluate the exact log-density of a directional-Gamma mixture."""
    (
        means,
        weights,
        covariances,
        latent_directions,
        gamma_shape,
    ) = _validate_parameters(
        means,
        weights,
        covariances,
        latent_directions,
        gamma_shape,
    )
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != means.shape[1]:
        raise ValueError("X and means have incompatible shapes")

    log_components = np.empty((len(weights), X.shape[0]), dtype=float)
    for component in range(len(weights)):
        log_components[component] = (
            np.log(weights[component])
            + log_directional_gamma_component_pdf(
                X,
                means[component],
                covariances[component],
                latent_directions[component],
                gamma_shape,
            )
        )
    return logsumexp(log_components, axis=0)
