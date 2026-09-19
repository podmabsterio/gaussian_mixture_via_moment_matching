import numpy as np

from experiments.synthetic_data.gaussian.sampler_core import GaussianDatasetGenerator
from experiments.wrappers import SklearnBayesianGaussianMixtureWrapper


def test_uniform_weight_generator_is_exact():
    dataset = GaussianDatasetGenerator(
        n_features=2,
        n_components=3,
        n_samples=60,
        covariance_type="spherical",
        variance_spread=1.0,
        means_mode="fixed",
        uniform_weights=True,
        min_mahalanobis_distance=2.0,
    ).generate(7)

    np.testing.assert_array_equal(dataset["true_weights"], np.full(3, 1.0 / 3.0))


def test_bayesian_gmm_wrapper_fits_spherical_mixture():
    dataset = GaussianDatasetGenerator(
        n_features=2,
        n_components=3,
        n_samples=120,
        covariance_type="spherical",
        variance_spread=1.0,
        means_mode="fixed",
        uniform_weights=True,
        min_mahalanobis_distance=3.0,
    ).generate(11)
    model = SklearnBayesianGaussianMixtureWrapper(
        n_components=3,
        max_iter=100,
        random_state=5,
    ).fit(**dataset)

    params = model.params_dict()
    assert params["means"].shape == (3, 2)
    assert params["weights"].shape == (3,)
    assert params["covariances"].shape == (3, 2, 2)
    np.testing.assert_allclose(params["weights"].sum(), 1.0)
    assert np.all(np.isfinite(params["covariances"]))
