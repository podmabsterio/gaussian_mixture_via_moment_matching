import numpy as np

from experiments.synthetic_data.gaussian.sample_covariances import generate_covariances
from experiments.synthetic_data.gaussian.sample_means import generate_centers
from experiments.synthetic_data.gaussian.sample_weights import generate_weights
from experiments.synthetic_data.gaussian.sample import sample_gm_data


def _normalize_mahalanobis_distance(
    centers,
    covariances,
    min_distance,
    weights,
):
    mean = weights @ centers
    d_min = np.inf

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            cov = (covariances[i] + covariances[j]) / 2
            diff = centers[i] - centers[j]
            d = np.sqrt(diff @ np.linalg.solve(cov, diff))
            d_min = min(d_min, d)

    scale = min_distance / d_min
    return mean + scale * (centers - mean)


class GaussianDatasetGenerator:
    """Generate data from a finite Gaussian mixture."""

    def __init__(
        self,
        n_features,
        n_components,
        n_samples=None,
        covariance_type="spherical",
        variance_spread=1.0,
        anisotropy=1.0,
        means_mode="fixed",
        weights_concentration=10.0,
        min_mahalanobis_distance=None,
    ):
        self.n_features = n_features
        self.n_components = n_components
        self.n_samples = n_samples
        self.covariance_type = covariance_type
        self.variance_spread = variance_spread
        self.anisotropy = anisotropy
        self.means_mode = means_mode
        self.weights_concentration = weights_concentration
        self.min_mahalanobis_distance = min_mahalanobis_distance

    def generate(self, seed, n_samples=None):
        covs = generate_covariances(
            n_components=self.n_components,
            n_features=self.n_features,
            covariance_type=self.covariance_type,
            variance_spread=self.variance_spread,
            anisotropy=self.anisotropy,
            random_state=seed,
        )

        means = generate_centers(
            n_components=self.n_components,
            n_features=self.n_features,
            mode=self.means_mode,
            random_state=seed,
        )

        weights = generate_weights(
            n_components=self.n_components,
            concentration=self.weights_concentration,
            random_state=seed,
        )

        if self.min_mahalanobis_distance is not None:
            means = _normalize_mahalanobis_distance(
                means, covs, self.min_mahalanobis_distance, weights
            )

        if n_samples is None:
            if self.n_samples is None:
                raise ValueError(
                    "n_samples is not specified in initialization nor in generate method"
                )
            n_samples = self.n_samples

        X, labels = sample_gm_data(means, weights, covs, n_samples, seed)

        dataset = {
            "true_means": means,
            "true_covariances": covs,
            "true_weights": weights,
            "true_labels": labels,
            "X": X,
        }

        return dataset


# Compatibility with configs written before the distribution-specific package
# layout was introduced.
DatasetGenerator = GaussianDatasetGenerator
