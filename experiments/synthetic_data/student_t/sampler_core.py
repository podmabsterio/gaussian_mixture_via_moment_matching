import numpy as np

from experiments.synthetic_data.gaussian import GaussianDatasetGenerator
from experiments.synthetic_data.student_t.sample import sample_student_t_mixture


class StudentTDatasetGenerator(GaussianDatasetGenerator):
    """Generate a mixture of heavy-tailed elliptical Student-t components.

    ``true_covariances`` are both the actual component covariances and the
    covariances of the moment-matched Gaussian target used by the existing
    quality metrics.
    """

    def __init__(self, degrees_of_freedom=5.0, **gaussian_kwargs):
        super().__init__(**gaussian_kwargs)
        self.degrees_of_freedom = float(degrees_of_freedom)
        if (
            not np.isfinite(self.degrees_of_freedom)
            or self.degrees_of_freedom <= 2
        ):
            raise ValueError(
                "degrees_of_freedom must be finite and greater than 2"
            )

    def generate(self, seed, n_samples=None):
        dataset = super().generate(seed, n_samples)
        dataset["X"] = sample_student_t_mixture(
            dataset["true_means"],
            dataset["true_covariances"],
            dataset["true_labels"],
            self.degrees_of_freedom,
            seed=seed,
        )
        dataset["evaluation_mask"] = np.ones(dataset["X"].shape[0], dtype=bool)
        dataset["distribution"] = "student_t"
        dataset["degrees_of_freedom"] = self.degrees_of_freedom
        return dataset
