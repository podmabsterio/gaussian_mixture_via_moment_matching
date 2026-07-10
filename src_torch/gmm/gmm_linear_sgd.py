import torch

from src.gmm.gmm import GaussianMomentModel


class UniformRandomDirectionSampler:
    def __init__(self, dimension):
        self.dimension = dimension

    def sample(self, num_samples, device=None, dtype=None):
        random_vectors = torch.randn(
            num_samples, self.dimension, device=device, dtype=dtype
        )
        norms = random_vectors.norm(dim=1, keepdim=True)
        return random_vectors / norms


class GaussianMomentModelLinearSGD(GaussianMomentModel):
    def __init__(
        self,
        centers_generator=None,
        bandwidth_generator=None,
        initializer=None,
        n_components=3,
        alternating=True,
        entropy_pen_lambda=None,
        precompute_Z=False,
        direction_sampler=None,
    ):
        super().__init__(
            centers_generator=centers_generator,
            bandwidth_generator=bandwidth_generator,
            initializer=initializer,
            n_components=n_components,
            alternating=alternating,
            entropy_pen_lambda=entropy_pen_lambda,
            precompute_Z=precompute_Z,
        )
        self.direction_sampler = (
            UniformRandomDirectionSampler(self.d)
            if direction_sampler is None
            else direction_sampler
        )

    def num_index_functions(self):
        return self.q * (self.d + 1)

    def _prepare_index_batch(self, index_batch):
        if index_batch is None:
            return torch.arange(
                self.num_index_functions(),
                device=self.centers.device,
                dtype=torch.long,
            )

        return torch.as_tensor(
            index_batch, device=self.centers.device, dtype=torch.long
        )

    def _decode_index_batch(self, index_batch):
        index_batch = self._prepare_index_batch(index_batch)
        features_per_center = self.d + 1
        center_idx = index_batch // features_per_center
        feature_idx = index_batch % features_per_center

        if torch.any(center_idx >= self.q):
            raise IndexError(
                "index_batch contains an index outside the linear test function range"
            )

        return center_idx, feature_idx

    def compute_Z_batch(self, X, index_batch=None):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)

        if index_batch is None:
            return self._compute_full_Z_batch(X)

        center_idx, feature_idx = self._decode_index_batch(index_batch)
        centers = self.centers[center_idx]
        s = self.s[:, center_idx]

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        dist2.clamp_min_(0.0)

        base_vals = torch.exp(-dist2[:, None, :] / (2.0 * s.square()[None, :, :]))
        base_sum = base_vals.sum(dim=0)

        is_base = feature_idx == 0
        coord_idx = (feature_idx - 1).clamp_min(0)
        x_coord = X[:, coord_idx]
        center_coord = centers.gather(1, coord_idx[:, None]).squeeze(1)

        weighted_x_sum = (base_vals * x_coord[:, None, :]).sum(dim=0)
        linear_sum = weighted_x_sum - base_sum * center_coord[None, :]

        Z = torch.where(is_base[None, :], base_sum, linear_sum)
        return Z.permute(1, 0).reshape(-1)

    def _compute_full_Z_batch(self, X):
        q, d = self.centers.shape
        B, q_s = self.s.shape
        assert q_s == q

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (self.centers * self.centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ self.centers.T)
        dist2.clamp_min_(0.0)

        base_vals = torch.exp(-dist2[:, None, :] / (2.0 * self.s.square()[None, :, :]))
        base_sum = base_vals.sum(dim=0)
        weighted_X_sum = torch.einsum("nd,nbq->bqd", X, base_vals)
        linear_sum = weighted_X_sum - base_sum[..., None] * self.centers[None, :, :]

        Z = torch.empty((q, d + 1, B), dtype=X.dtype, device=X.device)
        Z[:, 0, :] = base_sum.T
        Z[:, 1:, :] = linear_sum.permute(1, 2, 0)
        return Z

    def get_Z(self, index_batch=None):
        B, q = self.s.shape
        if self.precompute_Z:
            Z = self.Z.reshape(q * (self.d + 1), B)
            if index_batch is not None:
                index_batch = self._prepare_index_batch(index_batch)
                Z = Z[index_batch]
            return Z.reshape(-1)

        X = self.X
        n = X.shape[0]
        Z = self.compute_Z_batch(X, index_batch) / n
        return Z.reshape(-1)

    def psi(self, index_batch=None):
        if index_batch is None:
            return self._full_psi()

        center_idx, feature_idx = self._decode_index_batch(index_batch)
        centers = self.centers[center_idx]
        s = self.s[:, center_idx]
        m = self.m
        sigma = self.sigma

        L = center_idx.numel()
        K = m.shape[0]
        assert m.shape == (self.K, self.d)
        assert sigma.shape == (K,)

        c2 = (centers * centers).sum(dim=1, keepdim=True)
        m2 = (m * m).sum(dim=1).unsqueeze(0)
        dist2 = c2 + m2 - 2.0 * (centers @ m.T)
        dist2.clamp_min_(0.0)

        sigma2 = sigma.square()[None, None, :]
        s2 = s.square()[:, :, None]
        denom = s2 + sigma2

        base = (1.0 + sigma2 / s2).pow(-self.d / 2.0) * torch.exp(
            -dist2[None, :, :] / (2.0 * denom)
        )

        is_base = feature_idx == 0
        coord_idx = (feature_idx - 1).clamp_min(0)
        center_coord = centers.gather(1, coord_idx[:, None]).squeeze(1)
        m_coord = m[:, coord_idx].T
        diff_coord = m_coord - center_coord[:, None]

        linear = base * (s2 / denom) * diff_coord[None, :, :]
        out = torch.where(is_base[None, :, None], base, linear)
        return out.permute(1, 0, 2).reshape(L * self.s.shape[0], K)

    def _full_psi(self):
        centers = self.centers
        m = self.m
        sigma = self.sigma
        s = self.s

        q, d = centers.shape
        K = m.shape[0]
        B, q_s = s.shape
        assert q_s == q
        assert m.shape == (K, d)
        assert sigma.shape == (K,)

        c2 = (centers * centers).sum(dim=1, keepdim=True)
        m2 = (m * m).sum(dim=1).unsqueeze(0)
        dist2 = c2 + m2 - 2.0 * (centers @ m.T)
        dist2.clamp_min_(0.0)

        sigma2 = sigma.square()[None, None, :]
        s2 = s.square()[:, :, None]
        denom = s2 + sigma2

        base = (1.0 + sigma2 / s2).pow(-d / 2.0) * torch.exp(
            -dist2[None, :, :] / (2.0 * denom)
        )
        factor = s2 / denom
        diff = m[None, :, :] - centers[:, None, :]
        linear = (base * factor)[..., None] * diff[None, :, :, :]

        out = torch.empty((B, q, d + 1, K), dtype=base.dtype, device=base.device)
        out[:, :, 0, :] = base
        out[:, :, 1:, :] = linear.permute(0, 1, 3, 2)
        return out.permute(1, 2, 0, 3).reshape(q * (d + 1) * B, K)
