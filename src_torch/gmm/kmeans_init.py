from sklearn.cluster import KMeans
import torch


def _estimate_cluster_sigmas(X, labels, means, n_components):
    d = X.shape[1]

    global_sigma = X.std(dim=0, unbiased=False).mean().clamp_min(1e-6)
    sigmas = torch.empty(n_components, device=X.device)

    for k in range(n_components):
        cluster_points = X[labels == k]

        if cluster_points.shape[0] == 0:
            sigmas[k] = global_sigma
            continue

        diff = cluster_points - means[k]
        var_k = (diff.pow(2).sum(dim=1).mean() / d).clamp_min(1e-12)
        sigmas[k] = torch.sqrt(var_k)

    return sigmas.clamp_min(1e-6)


def build_kmeans_init(X, n_components):
    X_tensor = torch.as_tensor(X)
    X_np = X_tensor.detach().cpu().numpy()

    kmeans = KMeans(
        n_clusters=n_components,
        n_init=10,
        random_state=42,
    )
    labels_np = kmeans.fit_predict(X_np)
    centers_np = kmeans.cluster_centers_

    labels = torch.as_tensor(labels_np, device=X_tensor.device, dtype=torch.long)
    init_m = torch.as_tensor(centers_np, device=X_tensor.device)

    counts = torch.bincount(labels, minlength=n_components)
    init_mu = counts / counts.sum().clamp_min(1.0)

    init_sigma = _estimate_cluster_sigmas(X_tensor, labels, init_m)
    kmeans_s = init_sigma[labels]
    return init_mu, init_sigma, init_m, kmeans_s
