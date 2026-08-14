"""Dataset generator with controlled component-wise directional skewness."""

import numpy as np

from experiments.synthetic_data._utils import make_rng
from experiments.synthetic_data.directional_gamma.sample import (
    log_directional_gamma_mixture_pdf,
    sample_directional_gamma_mixture,
)
from experiments.synthetic_data.gaussian import GaussianDatasetGenerator


SKEWNESS_DIRECTION_MODES = frozenset({"random", "toward", "away"})


def _nearest_component_direction(means, component):
    differences = means - means[component]
    squared_distances = np.sum(differences * differences, axis=1)
    squared_distances[component] = np.inf
    nearest = int(np.argmin(squared_distances))
    direction = differences[nearest]
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("component means must not coincide")
    return direction / norm


def make_skewness_directions(
    means,
    weights,
    covariances,
    mode,
    *,
    seed=None,
):
    """Construct latent and physical skewness directions.

    ``toward`` points the long Gamma tail from each component toward the
    mixture barycenter; ``away`` reverses it. If a component happens to equal
    the barycenter, the nearest other component provides an unambiguous
    fallback. ``random`` draws independent uniform directions on the sphere.

    The requested directions live in data coordinates. They are mapped through
    the inverse Cholesky factor so that after applying the component covariance
    the physical skewness still points exactly in the requested direction.
    """
    means = np.asarray(means, dtype=float)
    weights = np.asarray(weights, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    mode = str(mode)
    if mode not in SKEWNESS_DIRECTION_MODES:
        choices = ", ".join(sorted(SKEWNESS_DIRECTION_MODES))
        raise ValueError(f"skewness_direction must be one of: {choices}")

    n_components, n_features = means.shape
    rng = make_rng(seed, salt=401)
    if mode == "random":
        requested = rng.normal(size=(n_components, n_features))
        requested /= np.linalg.norm(requested, axis=1, keepdims=True)
    else:
        barycenter = (weights / np.sum(weights)) @ means
        requested = np.empty_like(means)
        for component in range(n_components):
            direction = barycenter - means[component]
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 1e-14:
                direction = _nearest_component_direction(means, component)
            else:
                direction = direction / norm
            requested[component] = direction
        if mode == "away":
            requested *= -1.0

    latent = np.empty_like(requested)
    physical = np.empty_like(requested)
    for component in range(n_components):
        cholesky = np.linalg.cholesky(covariances[component])
        latent_direction = np.linalg.solve(cholesky, requested[component])
        latent_direction /= np.linalg.norm(latent_direction)
        latent[component] = latent_direction

        physical_direction = cholesky @ latent_direction
        physical[component] = physical_direction / np.linalg.norm(
            physical_direction
        )

    return latent, physical


class DirectionalGammaDatasetGenerator(GaussianDatasetGenerator):
    """Generate skewed components with exact configured means/covariances.

    ``gamma_shape`` controls standardized skewness, which equals
    ``2 / sqrt(gamma_shape)``. Smaller values therefore produce stronger
    asymmetry. The generated evaluation sample is independent of the training
    sample and is shared by every fitted model on a dataset realization.
    """

    def __init__(
        self,
        gamma_shape=2.0,
        skewness_direction="random",
        evaluation_num_samples=10_000,
        **gaussian_kwargs,
    ):
        super().__init__(**gaussian_kwargs)
        self.gamma_shape = float(gamma_shape)
        self.skewness_direction = str(skewness_direction)
        self.evaluation_num_samples = int(evaluation_num_samples)

        if not np.isfinite(self.gamma_shape) or self.gamma_shape <= 0:
            raise ValueError("gamma_shape must be finite and positive")
        if self.skewness_direction not in SKEWNESS_DIRECTION_MODES:
            choices = ", ".join(sorted(SKEWNESS_DIRECTION_MODES))
            raise ValueError(f"skewness_direction must be one of: {choices}")
        if self.evaluation_num_samples < 2:
            raise ValueError("evaluation_num_samples must be at least 2")

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples)
        latent_directions, physical_directions = make_skewness_directions(
            dataset["true_means"],
            dataset["true_weights"],
            dataset["true_covariances"],
            self.skewness_direction,
            seed=seed,
        )

        X, labels = sample_directional_gamma_mixture(
            dataset["true_means"],
            dataset["true_weights"],
            dataset["true_covariances"],
            latent_directions,
            self.gamma_shape,
            labels=dataset["true_labels"],
            seed=seed,
            salt=501,
        )
        evaluation_X, evaluation_labels = sample_directional_gamma_mixture(
            dataset["true_means"],
            dataset["true_weights"],
            dataset["true_covariances"],
            latent_directions,
            self.gamma_shape,
            num_samples=self.evaluation_num_samples,
            seed=seed,
            salt=601,
        )
        evaluation_true_log_pdf = log_directional_gamma_mixture_pdf(
            evaluation_X,
            dataset["true_means"],
            dataset["true_weights"],
            dataset["true_covariances"],
            latent_directions,
            self.gamma_shape,
        )

        dataset.update(
            {
                "X": X,
                "true_labels": labels,
                "evaluation_mask": np.ones(labels.size, dtype=bool),
                "evaluation_X": evaluation_X,
                "evaluation_labels": evaluation_labels,
                "evaluation_true_log_pdf": evaluation_true_log_pdf,
                "distribution": "directional_gamma",
                "gamma_shape": self.gamma_shape,
                "standardized_skewness": 2.0 / np.sqrt(self.gamma_shape),
                "skewness_direction_mode": self.skewness_direction,
                "latent_skewness_directions": latent_directions,
                "skewness_directions": physical_directions,
            }
        )
        return dataset
