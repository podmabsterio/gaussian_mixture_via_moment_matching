import numpy as np

from src_np.test_functions_response import compute_psi_one_s_full_covariance_gmm
from src_np.gmm import OneSGaussianMixtureModel


MIN_SIGMA = 1e-6


class PopulationWrapper(OneSGaussianMixtureModel):
    def __init__(self, means_noise_coef=0.0, sigmas_noise_coef=0.0, **kwargs):
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef

        super().__init__(**kwargs)

    def fit(self, true_means, true_weights, true_covariances, X, **kwargs):
        n_samples, d = X.shape

        def compute_population_Z_callback(s, test_centers, directions, **kwargs):
            population_Z = compute_psi_one_s_full_covariance_gmm(
                test_centers,
                directions,
                s,
                true_means,
                true_covariances,
                true_weights,
            )
            return population_Z

        rng = np.random.default_rng(1)
        means_noise = rng.normal(size=(self.k, d))
        sigma_noise = rng.normal(size=(self.k))

        self.means_ = true_means + self.means_noise_coef * means_noise

        true_spherical_variances = np.diagonal(
            true_covariances,
            axis1=1,
            axis2=2,
        ).mean(axis=1)
        true_spherical_sigmas = np.sqrt(true_spherical_variances)
        self.sigmas_ = true_spherical_sigmas + self.sigmas_noise_coef * sigma_noise
        self.sigmas_ = np.maximum(self.sigmas_, MIN_SIGMA)

        super().fit(
            X=X,
            Z_callback=compute_population_Z_callback,
        )
