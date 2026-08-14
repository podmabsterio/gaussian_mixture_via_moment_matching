import numpy as np
import pytest

from experiments.metrics import AdjustedRandIndex
from experiments.synthetic_data import (
    DirectionalGammaDatasetGenerator,
    GaussianDatasetGenerator,
    RadialOutlierDatasetGenerator,
    StudentTDatasetGenerator,
    UniformBackgroundDatasetGenerator,
    VarianceContaminationDatasetGenerator,
)
from experiments.synthetic_data.directional_gamma import (
    log_directional_gamma_mixture_pdf,
)
from experiments.synthetic_data.sampler_core import DatasetGenerator


BASE_KWARGS = {
    "n_features": 4,
    "n_components": 3,
    "n_samples": 200,
    "covariance_type": "full",
    "variance_spread": 3.0,
    "anisotropy": 4.0,
    "means_mode": "fixed",
    "weights_concentration": 10.0,
    "min_mahalanobis_distance": 3.0,
}


def test_old_gaussian_generator_name_remains_compatible():
    assert DatasetGenerator is GaussianDatasetGenerator


@pytest.mark.parametrize(
    "generator",
    [
        StudentTDatasetGenerator(degrees_of_freedom=6.0, **BASE_KWARGS),
        VarianceContaminationDatasetGenerator(
            contamination_fraction=0.1,
            variance_inflation=16.0,
            **BASE_KWARGS,
        ),
        UniformBackgroundDatasetGenerator(
            contamination_fraction=0.1,
            box_padding=4.0,
            **BASE_KWARGS,
        ),
        RadialOutlierDatasetGenerator(
            contamination_fraction=0.1,
            min_radius_scale=5.0,
            max_radius_scale=8.0,
            **BASE_KWARGS,
        ),
    ],
)
def test_non_gaussian_generators_are_reproducible(generator):
    first = generator.generate(7)
    second = generator.generate(7)

    for key in (
        "X",
        "true_means",
        "true_covariances",
        "true_weights",
        "true_labels",
        "evaluation_mask",
    ):
        np.testing.assert_array_equal(first[key], second[key])

    assert first["X"].shape == (BASE_KWARGS["n_samples"], BASE_KWARGS["n_features"])
    assert np.all(np.isfinite(first["X"]))


@pytest.mark.parametrize(
    "generator, labels_are_defined",
    [
        (
            VarianceContaminationDatasetGenerator(
                contamination_fraction=0.1,
                variance_inflation=16.0,
                **BASE_KWARGS,
            ),
            True,
        ),
        (
            UniformBackgroundDatasetGenerator(
                contamination_fraction=0.1,
                box_padding=4.0,
                **BASE_KWARGS,
            ),
            False,
        ),
        (
            RadialOutlierDatasetGenerator(
                contamination_fraction=0.1,
                min_radius_scale=5.0,
                max_radius_scale=8.0,
                **BASE_KWARGS,
            ),
            False,
        ),
    ],
)
def test_explicit_contamination_preserves_clean_target(
    generator,
    labels_are_defined,
):
    clean = GaussianDatasetGenerator(**BASE_KWARGS).generate(11)
    contaminated = generator.generate(11)
    mask = contaminated["contamination_mask"]

    assert np.sum(mask) == 20
    np.testing.assert_array_equal(contaminated["evaluation_mask"], ~mask)
    np.testing.assert_array_equal(contaminated["X"][~mask], clean["X"][~mask])
    np.testing.assert_array_equal(
        contaminated["true_means"],
        clean["true_means"],
    )
    np.testing.assert_array_equal(
        contaminated["true_covariances"],
        clean["true_covariances"],
    )
    np.testing.assert_array_equal(
        contaminated["true_weights"],
        clean["true_weights"],
    )

    if labels_are_defined:
        np.testing.assert_array_equal(
            contaminated["true_labels"],
            clean["true_labels"],
        )
    else:
        assert np.all(contaminated["true_labels"][mask] == -1)
        np.testing.assert_array_equal(
            contaminated["true_labels"][~mask],
            clean["true_labels"][~mask],
        )


def test_student_t_components_match_requested_first_two_moments():
    generator = StudentTDatasetGenerator(
        n_features=2,
        n_components=1,
        n_samples=50_000,
        covariance_type="full",
        variance_spread=1.0,
        anisotropy=3.0,
        means_mode="normal",
        weights_concentration=10.0,
        degrees_of_freedom=10.0,
    )
    dataset = generator.generate(13)

    empirical_mean = np.mean(dataset["X"], axis=0)
    empirical_covariance = np.cov(dataset["X"], rowvar=False, ddof=0)
    target_mean = dataset["true_means"][0]
    target_covariance = dataset["true_covariances"][0]

    np.testing.assert_allclose(empirical_mean, target_mean, atol=0.04)
    np.testing.assert_allclose(
        empirical_covariance,
        target_covariance,
        rtol=0.08,
        atol=0.03,
    )


def test_variance_contamination_inflates_tail_energy():
    generator = VarianceContaminationDatasetGenerator(
        n_features=3,
        n_components=1,
        n_samples=5_000,
        covariance_type="spherical",
        means_mode="normal",
        contamination_fraction=0.2,
        variance_inflation=25.0,
    )
    dataset = generator.generate(17)
    mask = dataset["contamination_mask"]
    residual = dataset["X"] - dataset["true_means"][0]
    squared_radius = np.sum(residual * residual, axis=1)

    assert np.mean(squared_radius[mask]) > 10 * np.mean(squared_radius[~mask])


def test_uniform_background_stays_inside_reported_box():
    generator = UniformBackgroundDatasetGenerator(
        contamination_fraction=0.15,
        box_padding=3.0,
        **BASE_KWARGS,
    )
    dataset = generator.generate(19)
    background = dataset["X"][dataset["contamination_mask"]]

    assert np.all(background >= dataset["background_lower"])
    assert np.all(background <= dataset["background_upper"])


def test_radial_outliers_stay_inside_reported_shell():
    generator = RadialOutlierDatasetGenerator(
        contamination_fraction=0.15,
        min_radius_scale=5.0,
        max_radius_scale=8.0,
        **BASE_KWARGS,
    )
    dataset = generator.generate(23)
    outliers = dataset["X"][dataset["contamination_mask"]]
    radii = np.linalg.norm(
        outliers - dataset["outlier_center"],
        axis=1,
    )

    assert np.all(radii >= dataset["outlier_min_radius"])
    assert np.all(radii <= dataset["outlier_max_radius"])


def test_adjusted_rand_index_ignores_masked_background_points():
    true_labels = np.array([0, 0, 1, 1, -1, -1])
    predicted_labels = np.array([1, 1, 0, 0, 0, 1])
    evaluation_mask = np.array([True, True, True, True, False, False])

    score = AdjustedRandIndex()(
        labels=predicted_labels,
        true_labels=true_labels,
        evaluation_mask=evaluation_mask,
    )

    assert score == pytest.approx(1.0)


def test_directional_gamma_components_match_requested_first_two_moments():
    generator = DirectionalGammaDatasetGenerator(
        n_features=3,
        n_components=1,
        n_samples=80_000,
        covariance_type="full",
        variance_spread=1.0,
        anisotropy=4.0,
        means_mode="normal",
        weights_concentration=10.0,
        gamma_shape=2.0,
        skewness_direction="random",
        evaluation_num_samples=100,
    )
    dataset = generator.generate(29)

    empirical_mean = np.mean(dataset["X"], axis=0)
    empirical_covariance = np.cov(dataset["X"], rowvar=False, ddof=0)
    np.testing.assert_allclose(
        empirical_mean,
        dataset["true_means"][0],
        atol=0.025,
    )
    np.testing.assert_allclose(
        empirical_covariance,
        dataset["true_covariances"][0],
        rtol=0.04,
        atol=0.025,
    )

    exact_log_pdf = log_directional_gamma_mixture_pdf(
        dataset["evaluation_X"],
        dataset["true_means"],
        dataset["true_weights"],
        dataset["true_covariances"],
        dataset["latent_skewness_directions"],
        dataset["gamma_shape"],
    )
    np.testing.assert_allclose(exact_log_pdf, dataset["evaluation_true_log_pdf"])
    assert np.all(np.isfinite(exact_log_pdf))


@pytest.mark.parametrize("mode, expected_sign", [("toward", 1.0), ("away", -1.0)])
def test_directional_gamma_skewness_points_relative_to_mixture(mode, expected_sign):
    generator = DirectionalGammaDatasetGenerator(
        n_features=4,
        n_components=3,
        n_samples=200,
        covariance_type="full",
        variance_spread=2.0,
        anisotropy=3.0,
        means_mode="fixed",
        weights_concentration=20.0,
        min_mahalanobis_distance=4.0,
        gamma_shape=2.0,
        skewness_direction=mode,
        evaluation_num_samples=100,
    )
    dataset = generator.generate(31)
    barycenter = dataset["true_weights"] @ dataset["true_means"]

    for component, direction in enumerate(dataset["skewness_directions"]):
        toward = barycenter - dataset["true_means"][component]
        toward /= np.linalg.norm(toward)
        alignment = np.dot(direction, toward)
        assert expected_sign * alignment > 1.0 - 1e-12


def test_directional_gamma_evaluation_sample_is_independent_and_reproducible():
    kwargs = {
        **BASE_KWARGS,
        "gamma_shape": 3.0,
        "skewness_direction": "random",
        "evaluation_num_samples": 250,
    }
    first = DirectionalGammaDatasetGenerator(**kwargs).generate(37)
    second = DirectionalGammaDatasetGenerator(**kwargs).generate(37)

    for key in (
        "X",
        "true_labels",
        "evaluation_X",
        "evaluation_labels",
        "evaluation_true_log_pdf",
        "skewness_directions",
        "latent_skewness_directions",
    ):
        np.testing.assert_array_equal(first[key], second[key])

    assert first["evaluation_X"].shape == (250, BASE_KWARGS["n_features"])
    assert not np.array_equal(
        first["X"][: min(len(first["X"]), 250)],
        first["evaluation_X"][: min(len(first["X"]), 250)],
    )
