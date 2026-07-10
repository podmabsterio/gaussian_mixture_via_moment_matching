import torch

EPS = 1e-12


def _build_bandwidth_multipliers(
    bandwidth_count,
    s_alpha,
    s_beta,
    *,
    device,
    dtype,
):
    if s_alpha < s_beta:
        raise ValueError("s_alpha must be >= s_beta")

    if bandwidth_count == 1:
        value = torch.sqrt(torch.tensor(s_alpha * s_beta, device=device, dtype=dtype))
        return value.view(1)

    log_multipliers = torch.linspace(
        torch.log(torch.tensor(s_beta, device=device, dtype=dtype)),
        torch.log(torch.tensor(s_alpha, device=device, dtype=dtype)),
        steps=bandwidth_count,
        device=device,
        dtype=dtype,
    )
    return torch.exp(log_multipliers)


def _count_nearby_with_kernel(
    centers,
    X,
    s,
    *,
    center_batch_size=256,
    candidate_batch_size=16,
):
    counts = torch.zeros(s.shape[0], device=X.device, dtype=X.dtype)
    x2 = (X * X).sum(dim=1, keepdim=True)

    for start in range(0, centers.shape[0], center_batch_size):
        centers_batch = centers[start : start + center_batch_size]
        c2 = (centers_batch * centers_batch).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ centers_batch.T)
        dist2.clamp_min_(0.0)

        for s_start in range(0, s.shape[0], candidate_batch_size):
            s_batch = s[s_start : s_start + candidate_batch_size]
            s2 = s_batch.square().view(1, -1, 1).clamp_min(EPS)
            values = torch.exp(-dist2[:, None, :] / (2.0 * s2))
            counts[s_start : s_start + s_batch.shape[0]] += values.sum(dim=(0, 2))

    return counts / centers.shape[0]


class BaseBandwidthGenerator:
    def __init__(self):
        pass

    def generate(self, X, centers, init_s=None, n_clusters=None):
        raise NotImplementedError()


class FixedBandwidthGenerator(BaseBandwidthGenerator):
    def __init__(self, bandwidth_count, s_alpha, s_beta):
        super().__init__()
        self.bandwidth_count = bandwidth_count
        self.s_alpha = s_alpha
        self.s_beta = s_beta

    def generate(self, X, centers, init_s=None, n_clusters=None):
        multipliers = _build_bandwidth_multipliers(
            bandwidth_count=self.bandwidth_count,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            device=X.device,
            dtype=X.dtype,
        )
        return multipliers[:, None].expand(-1, centers.shape[0])


class DimensionAwareBandwidthGenerator(BaseBandwidthGenerator):
    def __init__(
        self,
        bandwidth_count,
        s_alpha,
        s_beta,
        dimension_power=0.5,
        effective_dimension=None,
    ):
        super().__init__()
        self.bandwidth_count = bandwidth_count
        self.s_alpha = s_alpha
        self.s_beta = s_beta
        self.dimension_power = dimension_power
        self.effective_dimension = effective_dimension

    def generate(self, X, centers, init_s=None, n_clusters=None):
        multipliers = _build_bandwidth_multipliers(
            bandwidth_count=self.bandwidth_count,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            device=X.device,
            dtype=X.dtype,
        )
        dimension = (
            X.shape[1] if self.effective_dimension is None else self.effective_dimension
        )
        dimension_scale = torch.as_tensor(
            dimension,
            device=X.device,
            dtype=X.dtype,
        ).pow(self.dimension_power)
        return (dimension_scale * multipliers)[:, None].expand(-1, centers.shape[0])


class AdaptiveBandwidthGenerator(BaseBandwidthGenerator):
    def __init__(
        self,
        bandwidth_count=3,
        s_alpha=2.0,
        s_beta=0.5,
        k_neighbors=10,
        use_median=True,
        eps=EPS,
    ):
        super().__init__()
        self.bandwidth_count = bandwidth_count
        self.s_alpha = s_alpha
        self.s_beta = s_beta
        self.k_neighbors = k_neighbors
        self.use_median = use_median
        self.eps = eps

    def generate(self, X, centers, init_s=None, n_clusters=None):
        if init_s is None:
            n = X.shape[0]
            k = min(self.k_neighbors, n)

            dist2 = ((centers[:, None, :] - X[None, :, :]) ** 2).sum(dim=-1)
            knn_dist2, _ = torch.topk(dist2, k=k, dim=1, largest=False)
            knn_dist = torch.sqrt(knn_dist2.clamp_min(0.0) + self.eps)

            if self.use_median:
                local_scale = knn_dist.median(dim=1).values
            else:
                local_scale = knn_dist.mean(dim=1)
        else:
            local_scale = init_s

        local_scale = local_scale.clamp_min(self.eps)

        multipliers = _build_bandwidth_multipliers(
            bandwidth_count=self.bandwidth_count,
            s_alpha=self.s_alpha,
            s_beta=self.s_beta,
            device=X.device,
            dtype=X.dtype,
        )

        return multipliers[:, None] * local_scale[None, :]


class CountBandwidthGenerator(BaseBandwidthGenerator):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns

    def generate(self, X, centers, init_s=None, n_clusters=None):
        if n_clusters is None:
            raise ValueError("n_clusters must be provided")

        if isinstance(self.ns, int):
            c = torch.linspace(2.0, 1.0, self.ns)
            ns = X.shape[0] / (c * n_clusters)
        elif isinstance(self.ns, (list, tuple)):
            ns = torch.as_tensor(self.ns, device=X.device, dtype=X.dtype)
        else:
            raise ValueError("ns must be an int or a list/tuple of values")

        d = X.shape[1]

        s_candidates = torch.linspace(
            0.1,
            10.0 * d**0.5,
            100,
            device=X.device,
            dtype=X.dtype,
        )

        ns = torch.as_tensor(ns, device=X.device, dtype=X.dtype)

        if ns.numel() == 0:
            return torch.empty(
                0,
                centers.shape[0],
                device=X.device,
                dtype=X.dtype,
            )

        counts = _count_nearby_with_kernel(centers, X, s=s_candidates)
        mask = counts[:, None] >= ns[None, :]
        found = mask.any(dim=0)
        first_idx = mask.to(torch.long).argmax(dim=0)

        s = s_candidates[first_idx[found]]

        return s.unsqueeze(1).expand(-1, centers.shape[0])
