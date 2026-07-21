import numpy as np

from src_np.gmm import OneSGaussianMixtureModel


MIN_SIGMA = 1e-6


class OracleInitWrapper(OneSGaussianMixtureModel):
    def __init__(self, means_noise_coef=0.0, sigmas_noise_coef=0.0, **kwargs):
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef

        super().__init__(**kwargs)

    def fit(self, true_means, true_covariances, X, **kwargs):
        _, d = X.shape
        rng = np.random.default_rng(self.random_state)

        means_noise = rng.normal(size=(self.k, d))
        sigma_noise = rng.normal(size=self.k)

        self.means_ = true_means + self.means_noise_coef * means_noise

        true_spherical_variances = np.diagonal(
            true_covariances,
            axis1=1,
            axis2=2,
        ).mean(axis=1)
        true_spherical_sigmas = np.sqrt(true_spherical_variances)
        self.sigmas_ = true_spherical_sigmas + self.sigmas_noise_coef * sigma_noise
        self.sigmas_ = np.maximum(self.sigmas_, MIN_SIGMA)

        super().fit(X=X)
