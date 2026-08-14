import numpy as np

from experiments.synthetic_data.gaussian import GaussianDatasetGenerator
from experiments.synthetic_data.radial_outliers.sample import (
    contaminate_with_radial_outliers,
)


class RadialOutlierDatasetGenerator(GaussianDatasetGenerator):
    """Contaminate a clean GMM with gross outliers in a remote shell.

    This is the most direct bounded-influence stress test: contaminated points
    lie well beyond every component center. They have no component identity and
    are excluded from clustering metrics, while every estimator is fitted to
    the complete contaminated sample.
    """

    def __init__(
        self,
        contamination_fraction=0.05,
        min_radius_scale=6.0,
        max_radius_scale=10.0,
        **gaussian_kwargs,
    ):
        super().__init__(**gaussian_kwargs)
        self.contamination_fraction = float(contamination_fraction)
        self.min_radius_scale = float(min_radius_scale)
        self.max_radius_scale = float(max_radius_scale)
        if (
            not np.isfinite(self.contamination_fraction)
            or not 0 <= self.contamination_fraction < 1
        ):
            raise ValueError(
                "contamination_fraction must be finite and in [0, 1)"
            )
        if (
            not np.isfinite(self.min_radius_scale)
            or not np.isfinite(self.max_radius_scale)
            or self.min_radius_scale <= 0
            or self.max_radius_scale <= self.min_radius_scale
        ):
            raise ValueError(
                "radius scales must be finite and satisfy 0 < min < max"
            )

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples)
        X, mask, center, min_radius, max_radius = (
            contaminate_with_radial_outliers(
                dataset["X"],
                dataset["true_means"],
                dataset["true_weights"],
                dataset["true_covariances"],
                self.contamination_fraction,
                self.min_radius_scale,
                self.max_radius_scale,
                seed=seed,
            )
        )
        labels = dataset["true_labels"].copy()
        labels[mask] = -1

        dataset["X"] = X
        dataset["true_labels"] = labels
        dataset["contamination_mask"] = mask
        dataset["evaluation_mask"] = ~mask
        dataset["distribution"] = "radial_outliers"
        dataset["contamination_fraction"] = float(np.mean(mask))
        dataset["outlier_center"] = center
        dataset["outlier_min_radius"] = min_radius
        dataset["outlier_max_radius"] = max_radius
        return dataset
