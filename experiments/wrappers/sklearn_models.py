import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


MIN_VARIANCE = 1e-6


def oracle_noisy_spherical_init(
    true_means,
    true_weights,
    true_covariances,
    *,
    means_noise_coef=0.0,
    sigmas_noise_coef=0.0,
    random_state=None,
):
    true_means = np.asarray(true_means, dtype=float)
    true_weights = np.asarray(true_weights, dtype=float)
    true_covariances = np.asarray(true_covariances, dtype=float)

    rng = np.random.default_rng(random_state)
    n_components, n_features = true_means.shape

    means = true_means + means_noise_coef * rng.normal(
        size=(n_components, n_features)
    )
    variances = np.diagonal(true_covariances, axis1=1, axis2=2).mean(axis=1)
    sigmas = np.sqrt(variances) + sigmas_noise_coef * rng.normal(size=n_components)
    variances = np.maximum(sigmas * sigmas, MIN_VARIANCE)
    weights = true_weights / np.sum(true_weights)

    return means, weights, variances


def spherical_variances_to_covariances(variances, n_features):
    return variances[:, None, None] * np.eye(n_features)


def kmeans_spherical_init(
    X,
    n_components,
    *,
    n_init=10,
    max_iter=300,
    random_state=None,
):
    X = np.asarray(X, dtype=float)
    n_components = int(n_components)

    estimator = KMeans(
        n_clusters=n_components,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm="lloyd",
    )
    labels = estimator.fit_predict(X)
    means = estimator.cluster_centers_.astype(float)

    counts = np.bincount(labels, minlength=n_components).astype(float)
    weights = counts / np.maximum(counts.sum(), 1.0)

    n_features = X.shape[1]
    global_variance = np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))
    global_variance = max(global_variance / n_features, MIN_VARIANCE)

    variances = np.empty(n_components, dtype=float)
    for component_id in range(n_components):
        mask = labels == component_id
        if np.any(mask):
            diff = X[mask] - means[component_id]
            variances[component_id] = np.mean(np.sum(diff * diff, axis=1))
            variances[component_id] /= n_features
        else:
            variances[component_id] = global_variance
        variances[component_id] = max(variances[component_id], MIN_VARIANCE)

    weights = np.maximum(weights, 1e-12)
    weights /= weights.sum()

    return means, weights, variances, labels, estimator


class SklearnGaussianMixtureWrapper:
    def __init__(
        self,
        n_components,
        init_mode="oracle",
        means_noise_coef=0.0,
        sigmas_noise_coef=0.0,
        covariance_type="spherical",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        kmeans_n_init=10,
        kmeans_max_iter=300,
        random_state=None,
    ):
        if covariance_type != "spherical":
            raise ValueError("Only covariance_type='spherical' is supported for now")
        if init_mode not in {"oracle", "kmeans"}:
            raise ValueError("init_mode must be 'oracle' or 'kmeans'")
        self.n_components = int(n_components)
        self.init_mode = init_mode
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = int(max_iter)
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.random_state = random_state

    def fit(self, true_means, true_weights, true_covariances, X, **kwargs):
        if self.init_mode == "oracle":
            init_means, init_weights, init_variances = oracle_noisy_spherical_init(
                true_means,
                true_weights,
                true_covariances,
                means_noise_coef=self.means_noise_coef,
                sigmas_noise_coef=self.sigmas_noise_coef,
                random_state=self.random_state,
            )
        else:
            init_means, init_weights, init_variances, _, self.initial_kmeans_ = (
                kmeans_spherical_init(
                    X,
                    self.n_components,
                    n_init=self.kmeans_n_init,
                    max_iter=self.kmeans_max_iter,
                    random_state=self.random_state,
                )
            )

        self.estimator_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=1,
            init_params="random",
            weights_init=init_weights,
            means_init=init_means,
            precisions_init=1.0 / init_variances,
            random_state=self.random_state,
        )
        self.estimator_.fit(X)

        self.means_ = self.estimator_.means_
        self.weights_ = self.estimator_.weights_
        self.covariances_ = spherical_variances_to_covariances(
            self.estimator_.covariances_,
            self.means_.shape[1],
        )
        self.history_per_s = [np.asarray([-float(self.estimator_.lower_bound_)])]
        return self

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": self.covariances_,
        }


class SklearnKMeansWrapper:
    def __init__(
        self,
        n_components,
        init_mode="oracle",
        means_noise_coef=0.0,
        sigmas_noise_coef=0.0,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        if init_mode not in {"oracle", "kmeans"}:
            raise ValueError("init_mode must be 'oracle' or 'kmeans'")
        self.n_components = int(n_components)
        self.init_mode = init_mode
        self.means_noise_coef = means_noise_coef
        self.sigmas_noise_coef = sigmas_noise_coef
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = tol
        self.random_state = random_state

    def fit(self, true_means, true_weights, true_covariances, X, **kwargs):
        if self.init_mode == "oracle":
            init_means, _, init_variances = oracle_noisy_spherical_init(
                true_means,
                true_weights,
                true_covariances,
                means_noise_coef=self.means_noise_coef,
                sigmas_noise_coef=self.sigmas_noise_coef,
                random_state=self.random_state,
            )
            n_init = 1
        else:
            init_means, _, init_variances, _, self.initial_kmeans_ = (
                kmeans_spherical_init(
                    X,
                    self.n_components,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
            )
            n_init = 1

        self.estimator_ = KMeans(
            n_clusters=self.n_components,
            init=init_means,
            n_init=n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm="lloyd",
        )
        labels = self.estimator_.fit_predict(X)

        self.means_ = self.estimator_.cluster_centers_
        counts = np.bincount(labels, minlength=self.n_components).astype(float)
        self.weights_ = counts / np.maximum(counts.sum(), 1.0)

        n_features = X.shape[1]
        global_variance = np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))
        global_variance = max(global_variance / n_features, MIN_VARIANCE)

        variances = np.empty(self.n_components, dtype=float)
        for component_id in range(self.n_components):
            mask = labels == component_id
            if np.any(mask):
                diff = X[mask] - self.means_[component_id]
                variances[component_id] = np.mean(np.sum(diff * diff, axis=1))
                variances[component_id] /= n_features
            else:
                variances[component_id] = init_variances[component_id]
            variances[component_id] = max(variances[component_id], MIN_VARIANCE)

        if np.any(counts == 0):
            self.weights_ = np.maximum(self.weights_, 1e-12)
            self.weights_ /= self.weights_.sum()
            variances[counts == 0] = global_variance

        self.covariances_ = spherical_variances_to_covariances(
            variances,
            n_features,
        )
        self.history_per_s = [
            np.asarray([float(self.estimator_.inertia_) / max(X.shape[0], 1)])
        ]
        return self

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": self.covariances_,
        }
