import torch

from src.gmm.gmm import EPS, GaussianMomentModel
from src.gmm.gmm_linear_sgd import UniformRandomDirectionSampler


class GaussianMomentModelDirectionalSGD(GaussianMomentModel):
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
        num_directions=None,
        include_base_kernel=True,
        update_directions_on_loss=False,
        direction_update_Z_batch_size=256,
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
        self.direction_sampler = direction_sampler
        self.num_directions = num_directions
        self.include_base_kernel = include_base_kernel
        self.update_directions_on_loss = update_directions_on_loss
        self.direction_update_Z_batch_size = direction_update_Z_batch_size

    def insert_data(self, X, batch_size=16):
        self.init_params = self.initializer.get_initial_params(X)
        init_mu = self.init_params.get("init_mu")
        init_sigma = self.init_params.get("init_sigma")
        init_m = self.init_params.get("init_m")
        init_s = self.init_params.get("init_s", None)

        centers = self.centers_generator.generate(X)
        s = self.bandwidth_generator.generate(X, centers, init_s, n_clusters=self.K)
        q, d = centers.shape
        self.q = q
        self.d = d

        if self.num_directions is None:
            self.num_directions = d
        self.num_directions = int(self.num_directions)
        if self.num_directions <= 0:
            raise ValueError("num_directions must be positive")
        if self.direction_sampler is None:
            self.direction_sampler = UniformRandomDirectionSampler(d)

        self.register_buffer("centers", centers)
        self.register_buffer("s", s)
        self.register_buffer("X", X)
        self.register_buffer("directions", self._sample_directions())

        if self.precompute_Z:
            Z = self.compute_Z(X, batch_size=batch_size)
        else:
            Z = None

        self.register_buffer("Z", Z)
        self.register_buffer("z", Z.clone() if Z is not None else None)

        self.logits_mu = torch.nn.Parameter(torch.log(init_mu.clamp_min(1e-12)).clone())
        self.log_sigma = torch.nn.Parameter(torch.log(init_sigma.clone()))
        self.m = torch.nn.Parameter(init_m.clone())

    def _sample_directions(self):
        directions = self.direction_sampler.sample(
            self.q * self.num_directions,
            device=self.centers.device,
            dtype=self.centers.dtype,
        )
        expected_flat_shape = (self.q * self.num_directions, self.d)
        if tuple(directions.shape) != expected_flat_shape:
            raise ValueError(
                f"direction_sampler must return shape {expected_flat_shape}, got {tuple(directions.shape)}"
            )
        directions = directions.reshape(self.q, self.num_directions, self.d)
        return self._normalize_directions(directions)

    def _normalize_directions(self, directions):
        norms = directions.norm(dim=-1, keepdim=True).clamp_min(EPS)
        return directions / norms

    @torch.no_grad()
    def update_directions(self, directions=None, recompute_Z=True, batch_size=None):
        if directions is None:
            new_directions = self._sample_directions()
        else:
            new_directions = torch.as_tensor(
                directions,
                dtype=self.centers.dtype,
                device=self.centers.device,
            )
            expected_shape = (self.q, self.num_directions, self.d)
            if tuple(new_directions.shape) != expected_shape:
                raise ValueError(
                    f"directions must have shape {expected_shape}, got {tuple(new_directions.shape)}"
                )
            new_directions = self._normalize_directions(new_directions)

        self.directions.copy_(new_directions)

        if recompute_Z and self.precompute_Z:
            if self.X is None:
                raise RuntimeError(
                    "Cannot recompute Z after updating directions without stored data X"
                )
            if batch_size is None:
                batch_size = self.direction_update_Z_batch_size
            self.Z.copy_(self.compute_Z(self.X, batch_size=batch_size))
            self.z = self.Z.clone()

    def num_index_functions(self):
        return self.q * self._features_per_center()

    def _features_per_center(self):
        return self.num_directions + int(self.include_base_kernel)

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
        features_per_center = self._features_per_center()
        center_idx = index_batch // features_per_center
        feature_idx = index_batch % features_per_center

        if torch.any(center_idx >= self.q):
            raise IndexError(
                "index_batch contains an index outside the directional test function range"
            )

        if self.include_base_kernel:
            is_base = feature_idx == 0
            direction_idx = (feature_idx - 1).clamp_min(0)
        else:
            is_base = torch.zeros_like(feature_idx, dtype=torch.bool)
            direction_idx = feature_idx

        return center_idx, direction_idx, is_base

    def compute_Z_batch(self, X, index_batch=None):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)

        if index_batch is None:
            return self._compute_full_Z_batch(X)

        center_idx, direction_idx, is_base = self._decode_index_batch(index_batch)
        centers = self.centers[center_idx]
        directions = self.directions[center_idx, direction_idx]
        s = self.s[:, center_idx]

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        dist2.clamp_min_(0.0)

        base_vals = torch.exp(-dist2[:, None, :] / (2.0 * s.square()[None, :, :]))
        base_sum = base_vals.sum(dim=0)

        centered_projection = (
            (X[:, None, :] - centers[None, :, :])
            .mul(directions[None, :, :])
            .sum(dim=-1)
        )
        directional_sum = (base_vals * centered_projection[:, None, :]).sum(dim=0)

        Z = torch.where(is_base[None, :], base_sum, directional_sum)
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
        centered_sum = weighted_X_sum - base_sum[..., None] * self.centers[None, :, :]
        directional_sum = torch.einsum("bqd,qrd->bqr", centered_sum, self.directions)

        if self.include_base_kernel:
            Z = torch.empty(
                (q, self.num_directions + 1, B),
                dtype=X.dtype,
                device=X.device,
            )
            Z[:, 0, :] = base_sum.T
            Z[:, 1:, :] = directional_sum.permute(1, 2, 0)
            return Z

        return directional_sum.permute(1, 2, 0)

    def get_Z(self, index_batch=None):
        B, q = self.s.shape
        if self.precompute_Z:
            Z = self.Z.reshape(q * self._features_per_center(), B)
            if index_batch is not None:
                index_batch = self._prepare_index_batch(index_batch)
                Z = Z[index_batch]
            return Z.reshape(-1)

        n = self.X.shape[0]
        Z = self.compute_Z_batch(self.X, index_batch) / n
        return Z.reshape(-1)

    def psi(self, index_batch=None):
        if index_batch is None:
            return self._full_psi()

        center_idx, direction_idx, is_base = self._decode_index_batch(index_batch)
        centers = self.centers[center_idx]
        directions = self.directions[center_idx, direction_idx]
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

        diff_projection = (
            (m[None, :, :] - centers[:, None, :])
            .mul(directions[:, None, :])
            .sum(dim=-1)
        )
        directional = base * (s2 / denom) * diff_projection[None, :, :]
        out = torch.where(is_base[None, :, None], base, directional)
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
        diff_projection = torch.einsum("qkd,qrd->qrk", diff, self.directions)
        directional = (base * factor)[:, :, None, :] * diff_projection[None, :, :, :]

        if self.include_base_kernel:
            out = torch.empty(
                (B, q, self.num_directions + 1, K),
                dtype=base.dtype,
                device=base.device,
            )
            out[:, :, 0, :] = base
            out[:, :, 1:, :] = directional
            return out.permute(1, 2, 0, 3).reshape(q * (self.num_directions + 1) * B, K)

        return directional.permute(1, 2, 0, 3).reshape(q * self.num_directions * B, K)

    def loss(self, index_batch=None):
        if self.update_directions_on_loss:
            self.update_directions()

        return super().loss(index_batch)
