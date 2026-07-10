from src.gmm.gmm import GaussianMomentModel
from src.gmm.projectors import AffineOrthogonalProjectorParam

import torch
from torch import nn


class GaussianMomentModelProjection(GaussianMomentModel):
    def __init__(
        self,
        centers_generator=None,
        bandwidth_generator=None,
        initializer=None,
        n_components=3,
        alternating=True,
        entropy_pen_lambda=None,
        precompute_Z=False,
        projection_dim=2,
        rho=1.0,
        subspace_reg_lambda=1.0,
    ):
        if precompute_Z:
            # Z depends on the trainable projector, so cached empirical moments would
            # become stale immediately after the first optimizer step.
            precompute_Z = False
        super().__init__(
            centers_generator=centers_generator,
            bandwidth_generator=bandwidth_generator,
            initializer=initializer,
            n_components=n_components,
            alternating=alternating,
            entropy_pen_lambda=entropy_pen_lambda,
            precompute_Z=False,
        )
        self.projection_dim = projection_dim
        self.rho = rho
        self.subspace_reg_lambda = subspace_reg_lambda

    def insert_data(self, X, batch_size=16):
        super().insert_data(X, batch_size=batch_size)
        init_sigma = self.init_params.get("init_sigma")
        self.log_sigma_Q = nn.Parameter(torch.log(init_sigma.clone()))
        n = X.shape[-1]
        self.projector = AffineOrthogonalProjectorParam(
            n,
            self.projection_dim,
            dtype=X.dtype,
            device=X.device,
        )
        self.projector.initialize_by_points(self.init_params.get("init_m"))

    def _prepare_index_batch(self, index_batch):
        if index_batch is None:
            return None

        return torch.as_tensor(
            index_batch, device=self.centers.device, dtype=torch.long
        )

    def _select_centers_and_bandwidths(self, index_batch):
        centers = self.centers
        s = self.s

        index_batch = self._prepare_index_batch(index_batch)
        if index_batch is not None:
            centers = centers[index_batch]
            s = s[:, index_batch]

        return centers, s

    def compute_Z_batch(self, X, index_batch=None):
        X = torch.as_tensor(X, dtype=self.centers.dtype, device=self.centers.device)

        centers, s = self._select_centers_and_bandwidths(index_batch)
        q, d = centers.shape
        rho = torch.as_tensor(self.rho, dtype=X.dtype, device=X.device)

        B, q_s = s.shape
        assert q_s == q
        assert X.shape[1] == d

        # Project X and centers
        X_par, X_orth = self.projector(X)  # [N, d], [N, d]
        C_par, C_orth = self.projector(centers)  # [q, d], [q, d]

        # Squared distances in P-part
        x_par2 = (X_par * X_par).sum(dim=1, keepdim=True)  # [N, 1]
        c_par2 = (C_par * C_par).sum(dim=1).unsqueeze(0)  # [1, q]
        dist_par2 = x_par2 + c_par2 - 2.0 * (X_par @ C_par.T)  # [N, q]
        dist_par2.clamp_min_(0.0)

        # Squared distances in Q-part
        x_orth2 = (X_orth * X_orth).sum(dim=1, keepdim=True)  # [N, 1]
        c_orth2 = (C_orth * C_orth).sum(dim=1).unsqueeze(0)  # [1, q]
        dist_orth2 = x_orth2 + c_orth2 - 2.0 * (X_orth @ C_orth.T)  # [N, q]
        dist_orth2.clamp_min_(0.0)

        dist2 = dist_par2 + rho * dist_orth2  # [N, q]

        if B == 1:
            Z = torch.exp(-dist2 / (2.0 * s.square()[0])).sum(dim=0)  # [q]
            return Z

        s2 = s.square()[None, :, :]  # [1, B, q]
        Z = torch.exp(-dist2[:, None, :] / (2.0 * s2)).sum(dim=0)  # [B, q]
        return Z.permute(1, 0).reshape(q * B)

    @property
    def sigma_Q(self):
        return torch.exp(self.log_sigma_Q)

    def psi(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        m = self.m
        sigma_P = self.sigma
        sigma_Q = self.sigma_Q
        rho = torch.as_tensor(self.rho, dtype=centers.dtype, device=centers.device)

        q, d = centers.shape
        K = m.shape[0]
        B, q_s = s.shape

        assert q_s == q
        assert m.shape == (K, d)
        assert sigma_P.shape == (K,)
        assert sigma_Q.shape == (K,)

        # rank(P) and rank(Q)
        r = self.projection_dim
        d_minus_r = d - r

        # Project centers and means
        C_par, C_orth = self.projector(centers)  # [q, d], [q, d]
        M_par, M_orth = self.projector(m)  # [K, d], [K, d]

        # ||P(m - C)||^2  -> [q, K]
        c_par2 = (C_par * C_par).sum(dim=1, keepdim=True)  # [q, 1]
        m_par2 = (M_par * M_par).sum(dim=1).unsqueeze(0)  # [1, K]
        dist_par2 = c_par2 + m_par2 - 2.0 * (C_par @ M_par.T)  # [q, K]
        dist_par2.clamp_min_(0.0)

        # ||Q(m - C)||^2  -> [q, K]
        c_orth2 = (C_orth * C_orth).sum(dim=1, keepdim=True)  # [q, 1]
        m_orth2 = (M_orth * M_orth).sum(dim=1).unsqueeze(0)  # [1, K]
        dist_orth2 = c_orth2 + m_orth2 - 2.0 * (C_orth @ M_orth.T)  # [q, K]
        dist_orth2.clamp_min_(0.0)

        # Broadcast to [B, q, K]
        dist_par2 = dist_par2[None, :, :]  # [1, q, K]
        dist_orth2 = dist_orth2[None, :, :]  # [1, q, K]

        s2 = s.square()[:, :, None]  # [B, q, 1]
        sigma_P2 = sigma_P.square()[None, None, :]  # [1, 1, K]
        sigma_Q2 = sigma_Q.square()[None, None, :]  # [1, 1, K]

        pref_par = (1.0 + sigma_P2 / s2).pow(-r / 2.0)  # [B, q, K]
        pref_orth = (1.0 + rho * sigma_Q2 / s2).pow(-d_minus_r / 2.0)  # [B, q, K]

        exp_par = -dist_par2 / (2.0 * (s2 + sigma_P2))
        exp_orth = -rho * dist_orth2 / (2.0 * (s2 + rho * sigma_Q2))

        out = pref_par * pref_orth * torch.exp(exp_par + exp_orth)  # [B, q, K]
        out = out.permute(1, 0, 2).reshape(q * B, K)
        return out

    def _additional_regularization(self):
        _, orth = self.projector(self.m)
        residual = orth - self.projector.mean
        return self.subspace_reg_lambda * (residual**2).sum()

    def parameter_blocks(self):
        blocks = super().parameter_blocks()
        blocks["orthogonal_scales"] = [self.log_sigma_Q]
        blocks["scales"] = blocks["scales"] + blocks["orthogonal_scales"]
        blocks["projector"] = list(self.projector.parameters())
        blocks["geometry"] = blocks["means"] + blocks["projector"]
        blocks["component_params"] = blocks["means"] + blocks["scales"]
        blocks["all"] = (
            blocks["weights"] + blocks["component_params"] + blocks["projector"]
        )
        return blocks

    def fitted_isotropic_variances(self):
        _, d = self.m.shape
        r = self.projection_dim
        d_minus_r = d - r
        return (r * self.sigma.square() + d_minus_r * self.sigma_Q.square()) / d

    def fitted_covariances(self):
        _, d = self.m.shape
        projection = self.projector.projector_matrix()
        orthogonal_projection = (
            torch.eye(d, dtype=projection.dtype, device=projection.device) - projection
        )
        return (
            self.sigma.square()[:, None, None] * projection[None, :, :]
            + self.sigma_Q.square()[:, None, None] * orthogonal_projection[None, :, :]
        )

    def loss(self, index_batch=None):
        Z = self.get_Z(index_batch)
        M = self.M(index_batch)
        data_term = self._data_term(M, Z)
        self.z = self.get_z(M, Z)
        entropy_reg = self._entropy_term()
        additional_reg = self._additional_regularization()

        coef = 1.0
        if index_batch is not None:
            coef = self.num_index_functions() / len(index_batch)

        return coef * data_term + entropy_reg + additional_reg
