import numpy as np

from experiments.visualization import PCAProjector2D, project_dataset_to_2d


def test_projection_transforms_means_and_covariances_to_pca_coordinates():
    rng = np.random.default_rng(9)
    dataset = {
        "X": rng.normal(size=(80, 3)),
        "true_means": np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]),
        "true_covariances": np.array([np.eye(3), 4.0 * np.eye(3)]),
    }

    projection = project_dataset_to_2d(dataset)

    assert projection["X"].shape == (80, 2)
    assert projection["true_means"].shape == (2, 2)
    assert projection["true_covariances"].shape == (2, 2, 2)
    np.testing.assert_allclose(projection["true_covariances"][0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(projection["true_covariances"][1], 4.0 * np.eye(2), atol=1e-12)


def test_projection_embeds_one_dimensional_data_without_failing():
    dataset = {
        "X": np.array([[-2.0], [0.0], [2.0]]),
        "true_means": np.array([[-1.0], [1.0]]),
        "true_covariances": np.array([[[1.0]], [[4.0]]]),
    }

    projection = project_dataset_to_2d(dataset)

    np.testing.assert_allclose(projection["X"][:, 1], 0.0)
    np.testing.assert_allclose(projection["true_means"][:, 1], 0.0)
    np.testing.assert_allclose(projection["true_covariances"][:, 1, :], 0.0)
    np.testing.assert_allclose(projection["explained_variance_ratio"], [1.0, 0.0])


def test_fitted_projector_reuses_the_same_basis_for_estimated_geometry():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(50, 4))
    means = np.array([[1.0, 0.0, -1.0, 0.5], [-0.5, 1.0, 0.2, 2.0]])
    covariances = np.stack([np.eye(4), 2.0 * np.eye(4)])

    projector = PCAProjector2D(X)
    projected_once = projector.transform(means)
    projected_again = projector.transform(means)
    covariance_projection = projector.transform_covariances(covariances)

    np.testing.assert_array_equal(projected_once, projected_again)
    np.testing.assert_allclose(covariance_projection[0], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(covariance_projection[1], 2.0 * np.eye(2), atol=1e-12)
