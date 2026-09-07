from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from sklearn.decomposition import PCA


class PCAProjector2D:
    """Fit one reusable two-dimensional projection to observed data."""

    def __init__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self.n_features = X.shape[1]
        if self.n_features == 1:
            self.center_ = np.mean(X, axis=0)
            self.components_ = np.array([[1.0], [0.0]])
            self.explained_variance_ratio_ = np.array([1.0, 0.0])
        else:
            pca = PCA(n_components=2)
            pca.fit(X)
            self.center_ = pca.mean_.copy()
            self.components_ = pca.components_.copy()
            self.explained_variance_ratio_ = pca.explained_variance_ratio_.copy()

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError("values must match the fitted feature dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("projection values must contain only finite values")
        return (values - self.center_) @ self.components_.T

    def transform_covariances(self, covariances):
        covariances = np.asarray(covariances, dtype=float)
        if (
            covariances.ndim != 3
            or covariances.shape[1:] != (self.n_features, self.n_features)
        ):
            raise ValueError("covariances must match the fitted feature dimension")
        if not np.all(np.isfinite(covariances)):
            raise ValueError("covariances must contain only finite values")
        return np.asarray(
            [
                self.components_ @ covariance @ self.components_.T
                for covariance in covariances
            ]
        )


def project_dataset_to_2d(dataset):
    """Project samples and population component geometry into two dimensions.

    The returned covariances are expressed in the same coordinates as the
    projected observations and means, so callers can draw statistically
    meaningful standard-deviation ellipses without performing any additional
    dimensionality reduction.
    """
    X = np.asarray(dataset["X"], dtype=float)
    means = np.asarray(dataset["true_means"], dtype=float)
    covariances = np.asarray(dataset["true_covariances"], dtype=float)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("dataset X must be a non-empty two-dimensional array")
    n_components, n_features = means.shape
    if means.ndim != 2 or means.shape[1] != X.shape[1]:
        raise ValueError("true_means must match the feature dimension of X")
    if covariances.shape != (n_components, n_features, n_features):
        raise ValueError("true_covariances must match true_means")
    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(covariances))
    ):
        raise ValueError("dataset projection inputs must contain only finite values")

    projector = PCAProjector2D(X)
    X_2d = projector.transform(X)
    means_2d = projector.transform(means)
    covariances_2d = projector.transform_covariances(covariances)
    explained_variance_ratio = projector.explained_variance_ratio_

    return {
        "X": X_2d,
        "true_means": means_2d,
        "true_covariances": covariances_2d,
        "explained_variance_ratio": explained_variance_ratio,
    }


def plot_data_2d_projection(data_config, save_path, num_samples=3):
    import matplotlib.pyplot as plt

    dir_path = Path(save_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    dataset_generator = instantiate(data_config)

    for seed in range(num_samples):
        dataset = dataset_generator.generate(seed + 1)
        projection = project_dataset_to_2d(dataset)
        X_2d = projection["X"]
        means_2d = projection["true_means"]

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            ax.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=dataset.get("true_labels"),
                s=10,
                alpha=0.7,
                cmap="tab10",
            )
            ax.scatter(
                means_2d[:, 0],
                means_2d[:, 1],
                s=50,
                marker="x",
                c="red",
                linewidths=2,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Sample {seed + 1}")
            fig.tight_layout()
            fig.savefig(dir_path / f"sample_{seed + 1}.png", dpi=150)
        finally:
            plt.close(fig)
