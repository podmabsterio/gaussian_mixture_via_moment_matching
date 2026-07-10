import numpy as np
from sklearn.cluster import KMeans


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
