import numpy as np


def generate_centers(
    n_components,
    n_features,
    mode="normal",
    center_scale=1.0,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    if mode == "normal":
        return rng.normal(0, center_scale, size=(n_components, n_features))

    if mode == "fixed":
        if n_components > n_features + 1:
            raise ValueError("fixed mode requires n_components <= n_features + 1")

        centers = np.eye(n_components)
        centers -= centers.mean(axis=0)

        u, s, _ = np.linalg.svd(centers, full_matrices=False)
        centers = u[:, : n_components - 1] * s[: n_components - 1]

        out = np.zeros((n_components, n_features))
        out[:, : n_components - 1] = centers

        return center_scale * out
