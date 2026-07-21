from __future__ import annotations

import torch

from src_torch.gmm.gmm import EPS
from src_torch.gmm.gmm_dimension_free import GaussianMomentModelDimensionFree
from src_torch.gmm.gmm_linear_sgd import UniformRandomDirectionSampler


class GaussianMomentModelDimensionFreeLinear(GaussianMomentModelDimensionFree):
    def __init__(
        self,
        *args,
        direction_sampler=None,
        num_directions=None,
        include_base_kernel=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.direction_sampler = direction_sampler
        self.num_directions = num_directions
        self.include_base_kernel = include_base_kernel

    def _initialize_test_functions(self, X):
        if self.num_directions is None:
            self.num_directions = self.d
        self.num_directions = int(self.num_directions)
        if self.num_directions <= 0:
            raise ValueError("num_directions must be positive")
        if self.direction_sampler is None:
            self.direction_sampler = UniformRandomDirectionSampler(self.d)

        directions = self.direction_sampler.sample(
            self.q * self.num_directions,
            device=self.centers.device,
            dtype=self.centers.dtype,
        )
        expected_shape = (self.q * self.num_directions, self.d)
        if tuple(directions.shape) != expected_shape:
            raise ValueError(
                f"direction_sampler must return shape {expected_shape}, "
                f"got {tuple(directions.shape)}"
            )
        directions = directions.reshape(self.q, self.num_directions, self.d)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(EPS)
        self.register_buffer("directions", directions)

    def num_features(self):
        return self.num_directions + int(self.include_base_kernel)

    def compute_Z_batch(self, X, index_batch=None):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)
        centers, s = self._select_centers_and_bandwidths(index_batch)
        directions = self._select_directions(index_batch)

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        dist2.clamp_min_(0.0)

        base_values = torch.exp(-dist2[:, None, :] / (2.0 * s.square()[None, :, :]))
        base_sum = base_values.sum(dim=0)
        weighted_X_sum = torch.einsum("nd,nbq->bqd", X, base_values)
        centered_sum = weighted_X_sum - base_sum[..., None] * centers[None, :, :]
        directional_sum = torch.einsum("bqd,qrd->bqr", centered_sum, directions)

        if self.include_base_kernel:
            result = torch.empty(
                (centers.shape[0], s.shape[0], self.num_directions + 1),
                dtype=X.dtype,
                device=X.device,
            )
            result[:, :, 0] = base_sum.T
            result[:, :, 1:] = directional_sum.permute(1, 0, 2)
            return result

        return directional_sum.permute(1, 0, 2)

    def component_features(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        directions = self._select_directions(index_batch)
        base, directional = self._component_terms(centers, s, directions, full=False)
        return self._combine_features(base, directional)

    def full_component_features(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        directions = self._select_directions(index_batch)
        base, directional = self._component_terms(centers, s, directions, full=True)
        return self._combine_features(base, directional)

    def _select_directions(self, index_batch):
        if index_batch is None:
            return self.directions
        index_batch = torch.as_tensor(
            index_batch,
            device=self.centers.device,
            dtype=torch.long,
        )
        return self.directions[index_batch]

    def _component_terms(self, centers, s, directions, *, full):
        dist2 = self._center_mean_distances(centers)
        sigma2 = self.sigma.square()[None, None, :]
        s2 = s.square()[:, :, None]
        denominator = s2 + sigma2

        log_base = -dist2[None, :, :] / (2.0 * denominator)
        if full:
            log_base = log_base - self._dimension_log_coefficients()[:, None, :]
        base = torch.exp(log_base)

        difference = self.m[None, :, :] - centers[:, None, :]
        projected_difference = torch.einsum(
            "qkd,qrd->qrk",
            difference,
            directions,
        )
        directional = (base * s2 / denominator)[:, :, None, :] * projected_difference[
            None, :, :, :
        ]
        return base, directional

    def _combine_features(self, base, directional):
        if not self.include_base_kernel:
            return directional.permute(1, 0, 2, 3)

        result = torch.empty(
            (base.shape[1], base.shape[0], self.num_directions + 1, self.K),
            dtype=base.dtype,
            device=base.device,
        )
        result[:, :, 0, :] = base.permute(1, 0, 2)
        result[:, :, 1:, :] = directional.permute(1, 0, 2, 3)
        return result
