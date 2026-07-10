import numpy as np

from src_np.test_functions_response import compute_psi_one_s_isotropic_gmm
from src_np.gmm import OneSGaussianMixtureModel

# def compute_psi_one_s_isotropic_gmm(
#     test_centers,
#     test_directions,
#     s,
#     means,
#     sigmas,
#     weights=None,
#     amplitudes=None,
#     test_batch_size=8192,
# ):


class PopulationWrapper(OneSGaussianMixtureModel):
    def __init__(self, n_samples, d, means_noise_coef, sigmas_noise_coef, **kwargs):
        self.n_samples = n_samples
        self.d = d
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef

        super().__init__(**kwargs)

    def fit(self, true_means, true_weights, true_covariances, **kwargs):
        if true_covariances.ndim > 1:
            raise ValueError(
                "full dimensional covariance kernel convolution is not implemented"
            )

        def compute_population_Z_callback(s, test_centers, directions, **kwargs):
            population_Z = compute_psi_one_s_isotropic_gmm(
                test_centers,
                directions,
                s,
                true_means,
                true_covariances,
                true_weights,
            )
            return population_Z

        rng = np.random.default_rng(1)
        means_noise = rng.normal(size=(self.k, self.d))
        sigma_noise = rng.normal(size=(self.k))

        self.means_ = true_means + self.means_noise_coef * means_noise

        true_spherical_covariances = np.diagonal(
            true_covariances,
            axis1=1,
            axis2=2,
        ).mean(axis=1)
        self.sigmas_ = true_spherical_covariances + self.sigmas_noise_coef * sigma_noise

        super().fit(
            X=None,
            Z_callback=compute_population_Z_callback,
        )
