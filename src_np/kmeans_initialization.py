import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def initialize_data_point_gmm(
    data,
    n_components,
    random_state=None,
    min_sigma=1e-6,
):
    """Initialize component means from distinct observations.

    A shared nearest-neighbor scale avoids the degenerate singleton variances
    produced by K-means when ``n_components == n_samples``.
    """
    data = np.asarray(data, dtype=float)
    K = int(n_components)
    if (
        data.ndim != 2
        or data.shape[0] == 0
        or K < 1
        or K > data.shape[0]
        or not np.all(np.isfinite(data))
        or not np.isfinite(min_sigma)
        or min_sigma <= 0
    ):
        raise ValueError("invalid data-point initialization arguments")

    n, d = data.shape
    rng = np.random.default_rng(random_state)
    if K == n:
        indices = np.arange(n)
    else:
        indices = rng.choice(n, size=K, replace=False)

    global_center = np.mean(data, axis=0)
    global_sigma = np.sqrt(np.mean(np.sum((data - global_center) ** 2, axis=1)) / d)
    if n > 1:
        neighbor_count = min(n, 8)
        distances = NearestNeighbors(n_neighbors=neighbor_count).fit(data).kneighbors(
            data,
            return_distance=True,
        )[0]
        positive_distances = np.where(distances > 0, distances, np.inf)
        nearest_positive = np.min(positive_distances, axis=1)
        finite_distances = nearest_positive[np.isfinite(nearest_positive)]
        neighbor_sigma = (
            float(np.median(finite_distances)) / np.sqrt(d)
            if finite_distances.size
            else np.nan
        )
    else:
        neighbor_sigma = np.nan

    if not np.isfinite(neighbor_sigma) or neighbor_sigma <= 0:
        neighbor_sigma = global_sigma
    sigma = max(float(neighbor_sigma), float(min_sigma))
    weights = np.full(K, 1.0 / K, dtype=float)
    return {
        "means": data[indices].copy(),
        "sigmas": np.full(K, sigma, dtype=float),
        "weights": weights,
        "indices": indices,
    }


def initialize_dimension_free_one_s_gmm(
    data,
    n_components,
    s,
    random_state=None,
    n_init=20,
    max_iter=300,
    min_sigma=1e-6,
):
    data = np.asarray(data, dtype=float)
    s = float(s)
    K = int(n_components)

    assert (
        data.ndim == 2
        and K >= 1
        and K <= data.shape[0]
        and s > 0
        and n_init >= 1
        and max_iter >= 1
        and min_sigma > 0
    )

    n, d = data.shape

    km = KMeans(
        n_clusters=K,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(data)

    means = km.cluster_centers_.astype(float)
    weights = np.bincount(labels, minlength=K).astype(float) / n

    global_center = np.mean(data, axis=0)
    global_sigma = np.sqrt(np.mean(np.sum((data - global_center) ** 2, axis=1)) / d)
    global_sigma = max(global_sigma, min_sigma)

    sigmas = np.empty(K, dtype=float)

    for k in range(K):
        idx = labels == k

        if np.any(idx):
            diff = data[idx] - means[k]
            sigmas[k] = np.sqrt(np.mean(np.sum(diff * diff, axis=1)) / d)
        else:
            sigmas[k] = global_sigma

        sigmas[k] = max(sigmas[k], min_sigma)

    amplitudes = weights * np.exp(-0.5 * d * np.log1p((sigmas * sigmas) / (s * s)))

    return {
        "means": means,
        "sigmas": sigmas,
        "weights": weights,
        "amplitudes": amplitudes,
        "labels": labels,
        "kmeans": km,
    }
