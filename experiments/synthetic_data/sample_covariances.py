import numpy as np


def generate_covariances(
    n_components,
    n_features,
    covariance_type="full",
    base_variance=1.0,
    variance_spread=1.0,
    anisotropy=1.0,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    if n_components == 1:
        scales = np.array([base_variance])
    else:
        scales = base_variance * np.exp(
            np.linspace(
                -0.5 * np.log(variance_spread),
                0.5 * np.log(variance_spread),
                n_components,
            )
        )
        rng.shuffle(scales)

    def eigvals(scale):
        vals = np.geomspace(
            1 / np.sqrt(anisotropy),
            np.sqrt(anisotropy),
            n_features,
        )
        return scale * vals / vals.mean()

    def orthogonal():
        q, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
        return q

    covs = np.zeros((n_components, n_features, n_features))

    if covariance_type == "spherical":
        for k, s in enumerate(scales):
            covs[k] = s * np.eye(n_features)

    elif covariance_type == "diag":
        for k, s in enumerate(scales):
            vals = eigvals(s)
            rng.shuffle(vals)
            covs[k] = np.diag(vals)

    elif covariance_type == "full":
        for k, s in enumerate(scales):
            q = orthogonal()
            covs[k] = q @ np.diag(eigvals(s)) @ q.T

    elif covariance_type == "tied":
        q = orthogonal()
        cov = q @ np.diag(eigvals(base_variance)) @ q.T
        covs[:] = cov

    return covs
