import numpy as np

from src_np.test_functions_response import compute_psi_one_s_full_covariance_gmm
from src_np.gmm import MomentGaussianMixtureModel


MIN_SIGMA = 1e-6


class PopulationWrapper(MomentGaussianMixtureModel):
    def __init__(self, means_noise_coef=0.0, sigmas_noise_coef=0.0, **kwargs):
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef

        include_base_kernel = kwargs.pop("include_base_kernel", False)
        num_directions = kwargs.pop("num_directions", 1)
        joint_optimization = kwargs.pop("joint_optimization", False)
        if kwargs.get("init") is None:
            kwargs["init"] = "kmeans"
        kwargs.setdefault("n_zero_moments", int(include_base_kernel))
        kwargs.setdefault("n_first_moments", num_directions)
        kwargs.setdefault("n_second_moments", 0)
        kwargs.setdefault("normalize_moment_losses", False)
        kwargs.setdefault("scale_normalize_moments", False)
        kwargs.setdefault(
            "geometry_optimization",
            "joint" if joint_optimization else "component",
        )
        super().__init__(**kwargs)

    def fit(self, true_means, true_weights, true_covariances, X, **kwargs):
        n_samples, d = X.shape

        def compute_population_Z_callback(s, test_centers, test_directions, **kwargs):
            population_Z = compute_psi_one_s_full_covariance_gmm(
                test_centers,
                test_directions,
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
