from src_torch.gmm.gmm import GaussianMomentModel

import torch


class GaussianMomentModelLinear(GaussianMomentModel):
    def psi(self, index_batch):
        centers = self.centers
        m = self.m
        sigma = self.sigma
        s = self.s

        q, d = centers.shape
        K = m.shape[0]

        if s.ndim == 0:
            s = s.expand(1, q)
        elif s.ndim == 1:
            s = s[None, :]

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

        out = torch.empty((B, q, K, d + 1), dtype=base.dtype, device=base.device)
        out[..., 0] = base
        out[..., 1:] = linear

        return out.permute(1, 0, 3, 2).reshape(q * B * (d + 1), K)

    def compute_Z_batch(self, X, index_batch):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)

        n, d = X.shape
        q, d_centers = self.centers.shape
        assert d == d_centers

        s = self.s
        B, q_s = s.shape
        assert q_s == q

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (self.centers * self.centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ self.centers.T)
        dist2.clamp_min_(0.0)

        s2 = s.square()[None, :, :]
        base_vals = torch.exp(-dist2[:, None, :] / (2.0 * s2))

        base_sum = base_vals.sum(dim=0)
        weighted_X_sum = torch.einsum("nd,nbq->bqd", X, base_vals)
        linear_sum = weighted_X_sum - base_sum[..., None] * self.centers[None, :, :]

        Z = torch.cat([base_sum[..., None], linear_sum], dim=-1)
        return Z.permute(1, 0, 2).reshape(q * B * (d + 1))
