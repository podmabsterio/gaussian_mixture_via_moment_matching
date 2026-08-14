import numpy as np

from experiments.synthetic_data.gaussian import GaussianDatasetGenerator
from experiments.synthetic_data.uniform_background.sample import (
    contaminate_with_uniform_background,
)


class UniformBackgroundDatasetGenerator(GaussianDatasetGenerator):
    """Contaminate a clean GMM by diffuse uniform background observations.

    Background observations have no component identity. Their entries in
    ``true_labels`` are ``-1`` and ``evaluation_mask`` excludes them from
    clustering metrics. Parameter and density metrics continue to target the
    uncontaminated GMM.
    """

    def __init__(
        self,
        contamination_fraction=0.05,
        box_padding=4.0,
        **gaussian_kwargs,
    ):
        super().__init__(**gaussian_kwargs)
        self.contamination_fraction = float(contamination_fraction)
        self.box_padding = float(box_padding)
        if (
            not np.isfinite(self.contamination_fraction)
            or not 0 <= self.contamination_fraction < 1
        ):
            raise ValueError(
                "contamination_fraction must be finite and in [0, 1)"
            )
        if not np.isfinite(self.box_padding) or self.box_padding <= 0:
            raise ValueError("box_padding must be finite and positive")

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples)
        X, mask, lower, upper = contaminate_with_uniform_background(
            dataset["X"],
            dataset["true_means"],
            dataset["true_covariances"],
            self.contamination_fraction,
            self.box_padding,
            seed=seed,
        )
        labels = dataset["true_labels"].copy()
        labels[mask] = -1

        dataset["X"] = X
        dataset["true_labels"] = labels
        dataset["contamination_mask"] = mask
        dataset["evaluation_mask"] = ~mask
        dataset["distribution"] = "uniform_background"
        dataset["contamination_fraction"] = float(np.mean(mask))
        dataset["background_lower"] = lower
        dataset["background_upper"] = upper
        return dataset
