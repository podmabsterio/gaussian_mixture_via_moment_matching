from __future__ import annotations

import torch
from torch import nn

from src_torch.gmm.gmm import GaussianMomentModel
from src_torch.gmm.simplex_qp import solve_simplex_least_squares


LOSS_EUCLIDEAN = "euclidean"
LOSS_COSINE = "cosine"
LOSS_SCALE_WISE_COSINE = "scale_wise_cosine"
SUPPORTED_LOSSES = {LOSS_EUCLIDEAN, LOSS_COSINE, LOSS_SCALE_WISE_COSINE}


class GaussianMomentModelDimensionFree(GaussianMomentModel):
    def __init__(
        self,
        centers_generator=None,
        bandwidth_generator=None,
        initializer=None,
        n_components=3,
        alternating=False,
        entropy_pen_lambda=None,
        precompute_Z=True,
        loss_mode=LOSS_EUCLIDEAN,
        lambda_scale=0.0,
        scale_weights=None,
        cosine_eps=1e-12,
        relaxed_parametrization=False,
        relaxation_kappa=None,
    ):
        if alternating:
            raise ValueError(
                "Dimension-free models do not support alternating optimization"
            )
        if entropy_pen_lambda is not None:
            raise ValueError("Dimension-free models do not support entropy_pen_lambda")
        if loss_mode not in SUPPORTED_LOSSES:
            available = ", ".join(sorted(SUPPORTED_LOSSES))
            raise ValueError(
                f"Unknown loss_mode '{loss_mode}'. Available modes: {available}"
            )
        if lambda_scale < 0:
            raise ValueError("lambda_scale must be non-negative")
        if cosine_eps <= 0:
            raise ValueError("cosine_eps must be positive")
        if relaxed_parametrization and loss_mode == LOSS_SCALE_WISE_COSINE:
            raise ValueError(
                "relaxed_parametrization is only defined for euclidean and cosine losses"
            )
        if relaxation_kappa is not None and relaxation_kappa < 0:
            raise ValueError("relaxation_kappa must be non-negative")

        super().__init__(
            centers_generator=centers_generator,
            bandwidth_generator=bandwidth_generator,
            initializer=initializer,
            n_components=n_components,
            alternating=False,
            entropy_pen_lambda=None,
            precompute_Z=precompute_Z,
        )
        self.loss_mode = loss_mode
        self.lambda_scale = lambda_scale
        self.scale_weights_config = scale_weights
        self.cosine_eps = cosine_eps
        self.relaxed_parametrization = relaxed_parametrization
        self.relaxation_kappa = relaxation_kappa

    def insert_data(self, X, batch_size=16):
        super().insert_data(X, batch_size=batch_size)
        self._validate_common_bandwidths()

        initial_mu = self.init_params["init_mu"].clamp_min(1e-12)
        initial_mu = initial_mu / initial_mu.sum()
        log_mu = initial_mu.log()
        dimension_coefficients = self._dimension_log_coefficients().detach()

        del self.logits_mu
        if self.loss_mode == LOSS_SCALE_WISE_COSINE:
            self.c = nn.Parameter(log_mu[None, :] - dimension_coefficients)
        else:
            if self.relaxed_parametrization:
                self.reference_scale_index = int(self.bandwidth_values.argmin().item())
            else:
                self.reference_scale_index = int(self.bandwidth_values.argmax().item())
            self.c_ref = nn.Parameter(
                log_mu - dimension_coefficients[self.reference_scale_index]
            )

        self.register_buffer("_recovered_mu", initial_mu.detach().clone())
        self.register_buffer("scale_weights", self._build_scale_weights())

    def _validate_common_bandwidths(self):
        if self.s.ndim != 2 or self.s.shape[0] == 0:
            raise ValueError(
                "Dimension-free models require bandwidths with shape [B, q]"
            )
        if torch.any(self.s <= 0):
            raise ValueError("All bandwidths must be positive")

        bandwidth_values = self.s[:, 0]
        expected = bandwidth_values[:, None].expand_as(self.s)
        if not torch.allclose(self.s, expected):
            raise ValueError(
                "Dimension-free models require each bandwidth row to be constant "
                "across centers"
            )
        self.register_buffer("bandwidth_values", bandwidth_values.detach().clone())

    def _build_scale_weights(self):
        bandwidth_count = self.s.shape[0]
        if self.scale_weights_config is None:
            return torch.ones(
                bandwidth_count,
                dtype=self.s.dtype,
                device=self.s.device,
            )

        weights = torch.as_tensor(
            self.scale_weights_config,
            dtype=self.s.dtype,
            device=self.s.device,
        )
        if weights.shape != (bandwidth_count,):
            raise ValueError(
                f"scale_weights must have shape {(bandwidth_count,)}, "
                f"got {tuple(weights.shape)}"
            )
        if torch.any(weights <= 0):
            raise ValueError("scale_weights must be strictly positive")
        return weights

    def _dimension_log_coefficients(self):
        sigma2 = self.sigma.square()[None, :]
        s2 = self.bandwidth_values.square()[:, None]
        return 0.5 * self.d * torch.log1p(sigma2 / s2)

    def _relaxation_scale_offsets(self):
        reference_s2 = self.bandwidth_values[self.reference_scale_index].square()
        s2 = self.bandwidth_values.square()[:, None]
        sigma2 = self.sigma.square()[None, :]
        kappa = self.relaxation_kappa
        if kappa is None:
            kappa = self.d / (2.0 * reference_s2)
        return kappa * (1.0 - reference_s2 / s2) * sigma2

    def coefficients_by_scale(self):
        if self.loss_mode == LOSS_SCALE_WISE_COSINE:
            return self.c

        if self.relaxed_parametrization:
            return self.c_ref[None, :] + self._relaxation_scale_offsets()

        dimension_coefficients = self._dimension_log_coefficients()
        reference_coefficients = dimension_coefficients[self.reference_scale_index]
        return self.c_ref[None, :] + reference_coefficients - dimension_coefficients

    @property
    def mu(self):
        if self.loss_mode == LOSS_SCALE_WISE_COSINE:
            return self._recovered_mu

        reference_coefficient = self._dimension_log_coefficients()[
            self.reference_scale_index
        ]
        return torch.softmax(self.c_ref + reference_coefficient, dim=0)

    @property
    def amplitudes(self):
        coefficients = self.coefficients_by_scale()
        if self.loss_mode == LOSS_COSINE:
            coefficients = coefficients - coefficients.max().detach()
        elif self.loss_mode == LOSS_SCALE_WISE_COSINE:
            coefficients = (
                coefficients - coefficients.max(dim=1, keepdim=True).values.detach()
            )
        return torch.exp(coefficients)

    def parameter_blocks(self):
        weight_parameters = (
            [self.c] if self.loss_mode == LOSS_SCALE_WISE_COSINE else [self.c_ref]
        )
        blocks = {
            "weights": weight_parameters,
            "means": [self.m],
            "scales": [self.log_sigma],
        }
        blocks["component_params"] = blocks["means"] + blocks["scales"]
        blocks["all"] = blocks["weights"] + blocks["component_params"]
        return blocks

    def num_features(self):
        return 1

    def compute_Z_batch(self, X, index_batch=None):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)
        centers, s = self._select_centers_and_bandwidths(index_batch)

        x2 = (X * X).sum(dim=1, keepdim=True)
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)
        dist2.clamp_min_(0.0)

        values = torch.exp(-dist2[:, None, :] / (2.0 * s.square()[None, :, :]))
        return values.sum(dim=0).T.unsqueeze(-1)

    def component_features(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        dist2 = self._center_mean_distances(centers)
        denominator = s.square()[:, :, None] + self.sigma.square()[None, None, :]
        base = torch.exp(-dist2[None, :, :] / (2.0 * denominator))
        return base.permute(1, 0, 2).unsqueeze(2)

    def full_component_features(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        dist2 = self._center_mean_distances(centers)
        denominator = s.square()[:, :, None] + self.sigma.square()[None, None, :]
        log_base = -self._dimension_log_coefficients()[:, None, :] - dist2[
            None, :, :
        ] / (2.0 * denominator)
        return torch.exp(log_base).permute(1, 0, 2).unsqueeze(2)

    def _select_centers_and_bandwidths(self, index_batch):
        if index_batch is None:
            return self.centers, self.s

        index_batch = torch.as_tensor(
            index_batch,
            device=self.centers.device,
            dtype=torch.long,
        )
        return self.centers[index_batch], self.s[:, index_batch]

    def _center_mean_distances(self, centers):
        c2 = (centers * centers).sum(dim=1, keepdim=True)
        m2 = (self.m * self.m).sum(dim=1).unsqueeze(0)
        dist2 = c2 + m2 - 2.0 * (centers @ self.m.T)
        return dist2.clamp_min(0.0)

    def get_Z_structured(self, index_batch=None):
        if self.precompute_Z:
            if index_batch is None:
                return self.Z
            index_batch = torch.as_tensor(
                index_batch,
                device=self.centers.device,
                dtype=torch.long,
            )
            return self.Z[index_batch]

        return self.compute_Z_batch(self.X, index_batch) / self.X.shape[0]

    def get_Z(self, index_batch=None):
        return self.get_Z_structured(index_batch).reshape(-1)

    def M_structured(self, index_batch=None):
        features = self.component_features(index_batch)
        return torch.einsum("qbfk,bk->qbf", features, self.amplitudes)

    def M(self, index_batch=None):
        return self.M_structured(index_batch).reshape(-1)

    def _cosine_distance(self, observed, modeled):
        observed = observed.reshape(-1)
        modeled = modeled.reshape(-1)
        observed_norm = observed.norm()
        modeled_norm = modeled.norm()
        denominator = observed_norm * modeled_norm + self.cosine_eps
        return 1.0 - torch.dot(observed, modeled) / denominator

    def scale_consistency_penalty(self):
        if self.loss_mode != LOSS_SCALE_WISE_COSINE or self.lambda_scale == 0:
            return self.c.new_zeros(()) if hasattr(self, "c") else self.m.new_zeros(())

        corrected = self.c + self._dimension_log_coefficients()
        centered = corrected - corrected.mean(dim=1, keepdim=True)
        centered_mean = centered.mean(dim=0, keepdim=True)
        return self.lambda_scale * (centered - centered_mean).square().sum()

    def loss(self, index_batch=None):
        if self.loss_mode != LOSS_EUCLIDEAN and index_batch is not None:
            raise ValueError("Cosine dimension-free losses require full-batch moments")

        observed = self.get_Z_structured(index_batch)
        modeled = self.M_structured(index_batch)

        if self.loss_mode == LOSS_EUCLIDEAN:
            coefficient = 1.0
            if index_batch is not None:
                coefficient = self.num_index_functions() / len(index_batch)
            return coefficient * 0.5 * (observed - modeled).square().sum()

        if self.loss_mode == LOSS_COSINE:
            return self._cosine_distance(observed, modeled)

        per_scale = [
            self.scale_weights[scale_index]
            * self._cosine_distance(
                observed[:, scale_index, :],
                modeled[:, scale_index, :],
            )
            for scale_index in range(self.s.shape[0])
        ]
        return torch.stack(per_scale).sum() + self.scale_consistency_penalty()

    @torch.no_grad()
    def post_fit(self):
        if self.loss_mode != LOSS_SCALE_WISE_COSINE:
            return

        A = self.full_component_features().reshape(-1, self.K)
        z = self.get_Z_structured().reshape(-1)
        recovered = solve_simplex_least_squares(
            A,
            z,
            initial_weights=self._recovered_mu,
        )
        self._recovered_mu.copy_(recovered)
