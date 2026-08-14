import numpy as np

from experiments.synthetic_data.gaussian import GaussianDatasetGenerator
from experiments.synthetic_data.variance_contamination.sample import (
    contaminate_component_variances,
)


class VarianceContaminationDatasetGenerator(GaussianDatasetGenerator):
    """Generate a component-wise Gaussian scale-contamination mixture.

    A fraction of observations from component ``k`` is replaced by samples from
    ``N(mean_k, variance_inflation * covariance_k)``. Labels remain meaningful,
    but quality metrics target the uncontaminated Gaussian core. ARI is evaluated
    on the retained core observations through ``evaluation_mask``.
    """

    def __init__(
        self,
        contamination_fraction=0.05,
        variance_inflation=25.0,
        **gaussian_kwargs,
    ):
        super().__init__(**gaussian_kwargs)
        self.contamination_fraction = float(contamination_fraction)
        self.variance_inflation = float(variance_inflation)
        if (
            not np.isfinite(self.contamination_fraction)
            or not 0 <= self.contamination_fraction < 1
        ):
            raise ValueError(
                "contamination_fraction must be finite and in [0, 1)"
            )
        if (
            not np.isfinite(self.variance_inflation)
            or self.variance_inflation <= 1
        ):
            raise ValueError(
                "variance_inflation must be finite and greater than 1"
            )

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples)
        X, mask = contaminate_component_variances(
            dataset["X"],
            dataset["true_labels"],
            dataset["true_means"],
            dataset["true_covariances"],
            self.contamination_fraction,
            self.variance_inflation,
            seed=seed,
        )
        dataset["X"] = X
        dataset["contamination_mask"] = mask
        dataset["evaluation_mask"] = ~mask
        dataset["distribution"] = "variance_contamination"
        dataset["contamination_fraction"] = float(np.mean(mask))
        dataset["variance_inflation"] = self.variance_inflation
        dataset["population_covariances"] = (
            1.0
            + self.contamination_fraction * (self.variance_inflation - 1.0)
        ) * dataset["true_covariances"]
        return dataset
