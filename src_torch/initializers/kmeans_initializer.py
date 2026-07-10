from src.initializers.base_initializer import BaseInitializer

from sklearn.cluster import KMeans
import torch


class KMeansInitializer(BaseInitializer):
    def __init__(self, n_components, n_init=10, random_state=42):
        super().__init__(n_components)
        self.n_init = int(n_init)
        self.random_state = random_state

    def _estimate_cluster_sigmas(self, X, labels, means):
        d = X.shape[1]

        global_sigma = X.std(dim=0, unbiased=False).mean().clamp_min(1e-6)
        sigmas = torch.empty(self.n_components, device=X.device)

        for k in range(self.n_components):
            cluster_points = X[labels == k]

            if cluster_points.shape[0] == 0:
                sigmas[k] = global_sigma
                continue

            diff = cluster_points - means[k]
            var_k = (diff.pow(2).sum(dim=1).mean() / d).clamp_min(1e-12)
            sigmas[k] = torch.sqrt(var_k)

        return sigmas.clamp_min(1e-6)

    def get_initial_params(self, X):
        X_tensor = torch.as_tensor(X)
        X_np = X_tensor.detach().cpu().numpy()

        kmeans = KMeans(
            n_clusters=self.n_components,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels_np = kmeans.fit_predict(X_np)
        centers_np = kmeans.cluster_centers_

        labels = torch.as_tensor(labels_np, device=X_tensor.device, dtype=torch.long)
        init_m = torch.as_tensor(
            centers_np, device=X_tensor.device, dtype=X_tensor.dtype
        )

        counts = torch.bincount(labels, minlength=self.n_components)
        init_mu = counts.to(dtype=X_tensor.dtype) / counts.sum().clamp_min(1.0).to(
            dtype=X_tensor.dtype
        )

        init_sigma = self._estimate_cluster_sigmas(X_tensor, labels, init_m)
        kmeans_s = init_sigma[labels]

        return {
            "init_m": init_m,
            "init_sigma": init_sigma,
            "init_mu": init_mu,
            "init_s": kmeans_s,
        }


class NoisyKMeansInitializer(KMeansInitializer):
    def __init__(self, n_components, noise_fraction=0.1, min_relative_scale=1e-6):
        super().__init__(n_components)
        self.noise_fraction = noise_fraction
        self.min_relative_scale = min_relative_scale

    def _add_relative_noise(self, value):
        if self.noise_fraction <= 0.0:
            return value

        reference_scale = value.detach().abs().mean().clamp_min(self.min_relative_scale)
        scale = value.detach().abs().clamp_min(reference_scale)
        return value + self.noise_fraction * scale * torch.randn_like(value)

    def get_initial_params(self, X):
        params = super().get_initial_params(X)

        init_m = self._add_relative_noise(params["init_m"])
        init_sigma = self._add_relative_noise(params["init_sigma"]).clamp_min(1e-6)
        init_mu = self._add_relative_noise(params["init_mu"]).clamp_min(1e-12)
        init_mu = init_mu / init_mu.sum().clamp_min(1e-12)

        params["init_m"] = init_m
        params["init_sigma"] = init_sigma
        params["init_mu"] = init_mu

        if "init_s" in params and params["init_s"] is not None:
            params["init_s"] = self._add_relative_noise(params["init_s"]).clamp_min(
                1e-6
            )

        return params
