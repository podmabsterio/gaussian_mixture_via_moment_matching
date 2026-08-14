import numpy as np
import pytest

from experiments.metrics import (
    ComponentConditionalNLL,
    EnergyDistance,
    HeldOutNegativeLogLikelihood,
    SlicedWasserstein,
    TrueForwardKL,
    TrueForwardKLStandardError,
)
from experiments.metrics.metric_utils import log_gm_pdf
from experiments.synthetic_data.directional_gamma import (
    DirectionalGammaDatasetGenerator,
)
from experiments.synthetic_data.gaussian import sample_gm_data


def _skewed_dataset():
    return DirectionalGammaDatasetGenerator(
        n_features=3,
        n_components=2,
        n_samples=300,
        covariance_type="spherical",
        variance_spread=2.0,
        means_mode="fixed",
        weights_concentration=10.0,
        min_mahalanobis_distance=4.0,
        gamma_shape=2.0,
        skewness_direction="toward",
        evaluation_num_samples=4_000,
    ).generate(41)


def _true_gaussian_params(dataset):
    return {
        "means": dataset["true_means"],
        "weights": dataset["true_weights"],
        "covariances": dataset["true_covariances"],
    }


def test_true_kl_and_held_out_nll_share_the_expected_entropy_term():
    dataset = _skewed_dataset()
    params = _true_gaussian_params(dataset)

    nll = HeldOutNegativeLogLikelihood()(
        evaluation_X=dataset["evaluation_X"],
        **params,
    )
    kl = TrueForwardKL()(
        evaluation_X=dataset["evaluation_X"],
        evaluation_true_log_pdf=dataset["evaluation_true_log_pdf"],
        **params,
    )
    expected_kl = nll + np.mean(dataset["evaluation_true_log_pdf"])

    assert kl == pytest.approx(expected_kl)
    assert kl > 0.0
    assert TrueForwardKLStandardError()(
        evaluation_X=dataset["evaluation_X"],
        evaluation_true_log_pdf=dataset["evaluation_true_log_pdf"],
        **params,
    ) > 0.0


def test_component_conditional_nll_uses_matched_component_not_mixture_density():
    dataset = _skewed_dataset()
    params = _true_gaussian_params(dataset)
    score = ComponentConditionalNLL()(
        evaluation_X=dataset["evaluation_X"],
        evaluation_labels=dataset["evaluation_labels"],
        means=params["means"],
        covariances=params["covariances"],
    )
    mixture_nll = -np.mean(
        log_gm_pdf(dataset["evaluation_X"], **params)
    )

    assert np.isfinite(score)
    assert score != pytest.approx(mixture_nll)


def test_sample_distances_are_finite_reproducible_and_detect_large_shift():
    rng = np.random.default_rng(43)
    means = np.array([[-1.5, 0.0], [1.5, 0.0]])
    weights = np.array([0.5, 0.5])
    covariances = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    X, _ = sample_gm_data(means, weights, covariances, 500, seed=47)
    evaluation_X, _ = sample_gm_data(
        means,
        weights,
        covariances,
        1_500,
        seed=53,
    )
    skewness_directions = rng.normal(size=(2, 2))

    energy = EnergyDistance(seed=59)
    sliced = SlicedWasserstein(
        num_samples=1_000,
        num_random_projections=30,
        seed=61,
    )
    true_energy = energy(
        X=X,
        means=means,
        weights=weights,
        covariances=covariances,
    )
    true_sliced = sliced(
        evaluation_X=evaluation_X,
        skewness_directions=skewness_directions,
        means=means,
        weights=weights,
        covariances=covariances,
    )

    shifted_means = means + np.array([8.0, -5.0])
    shifted_energy = energy(
        X=X,
        means=shifted_means,
        weights=weights,
        covariances=covariances,
    )
    shifted_sliced = sliced(
        evaluation_X=evaluation_X,
        skewness_directions=skewness_directions,
        means=shifted_means,
        weights=weights,
        covariances=covariances,
    )

    assert np.isfinite(true_energy)
    assert np.isfinite(true_sliced)
    assert true_energy >= 0.0
    assert true_sliced >= 0.0
    assert shifted_energy > 3.0 * true_energy
    assert shifted_sliced > 3.0 * true_sliced
    assert energy(
        X=X,
        means=means,
        weights=weights,
        covariances=covariances,
    ) == pytest.approx(true_energy)
