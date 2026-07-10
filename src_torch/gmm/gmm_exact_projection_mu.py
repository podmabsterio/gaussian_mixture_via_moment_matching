import torch

from src.gmm.gmm import GaussianMomentModel
from src.gmm.projectors import AffineOrthogonalProjectorParam


class GaussianMomentModelExactProjectionMu(GaussianMomentModel):
    def __init__(
        self,
        centers_generator=None,
        bandwidth_generator=None,
        initializer=None,
        n_components=3,
        alternating=True,
        entropy_pen_lambda=None,
        precompute_Z=True,
        projection_dim=2,
        subspace_reg_lambda=1.0,
        train_projector=False,
        orthogonal_noise_scale=0.0,
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
        self.projection_dim = projection_dim
        self.subspace_reg_lambda = subspace_reg_lambda
        self.train_projector = train_projector
        self.orthogonal_noise_scale = orthogonal_noise_scale

    def insert_data(self, X, batch_size=16):
        super().insert_data(X, batch_size=batch_size)
        self._parameters["log_amplitudes"] = self._parameters.pop("logits_mu")

        s_value = self.s[0, 0]
        if self.s.shape[0] != 1 or torch.any(self.s != s_value):
            raise ValueError(
                "GaussianMomentModelExactProjectionMu only supports a single "
                "bandwidth value for all centers."
            )

        self.s_single_value = s_value

        if self.projection_dim > 0:
            scale = torch.as_tensor(
                self.d / self.projection_dim,
                dtype=self.log_sigma.dtype,
                device=self.log_sigma.device,
            )
            with torch.no_grad():
                self.log_sigma.add_(0.5 * scale.log())

        n = X.shape[-1]
        self.projector = AffineOrthogonalProjectorParam(
            n,
            self.projection_dim,
            dtype=X.dtype,
            device=X.device,
        )
        self.projector.initialize_by_points(X)

        with torch.no_grad():
            self.log_amplitudes.add_(self.log_psi_dimension_coef())

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

    def log_psi_dimension_coef(self):
        log_s = self.s_single_value.log()
        parallel_coef = self.projection_dim * (
            log_s - 0.5 * torch.logaddexp(2.0 * self.log_sigma, 2.0 * log_s)
        )
        d_minus_r = self.d - self.projection_dim
        orthogonal_noise = torch.as_tensor(
            self.orthogonal_noise_scale,
            dtype=self.log_sigma.dtype,
            device=self.log_sigma.device,
        )
        orthogonal_coef = (
            -0.5
            * d_minus_r
            * torch.log1p(orthogonal_noise.square() / self.s_single_value.square())
        )
        return parallel_coef + orthogonal_coef

    def psi(self, index_batch=None):
        centers, s = self._select_centers_and_bandwidths(index_batch)
        m = self.m
        sigma = self.sigma

        q, d = centers.shape
        K = m.shape[0]
        B, q_s = s.shape

        assert q_s == q
        assert m.shape == (K, d)
        assert sigma.shape == (K,)

        C_par, C_orth = self.projector(centers)
        M_par, M_orth = self.projector(m)
        orthogonal_noise = torch.as_tensor(
            self.orthogonal_noise_scale,
            dtype=centers.dtype,
            device=centers.device,
        )

        c_par2 = (C_par * C_par).sum(dim=1, keepdim=True)
        m_par2 = (M_par * M_par).sum(dim=1).unsqueeze(0)
        dist_par2 = c_par2 + m_par2 - 2.0 * (C_par @ M_par.T)
        dist_par2.clamp_min_(0.0)

        c_orth2 = (C_orth * C_orth).sum(dim=1, keepdim=True)
        m_orth2 = (M_orth * M_orth).sum(dim=1).unsqueeze(0)
        dist_orth2 = c_orth2 + m_orth2 - 2.0 * (C_orth @ M_orth.T)
        dist_orth2.clamp_min_(0.0)

        s2 = s.square()[:, :, None]
        sigma2 = sigma.square()[None, None, :]
        orthogonal_noise2 = orthogonal_noise.square()

        exp_par = -dist_par2[None, :, :] / (2.0 * (s2 + sigma2))
        exp_orth = -dist_orth2[None, :, :] / (2.0 * (s2 + orthogonal_noise2))

        out = torch.exp(exp_par + exp_orth)
        out = out.permute(1, 0, 2).reshape(q * B, K)
        return out

    def M(self, index_batch):
        psi_value = self.psi(index_batch)
        return psi_value @ self.amplitudes

    @property
    def mu(self):
        logits = self.log_amplitudes - self.log_psi_dimension_coef()
        normalized_weights = torch.softmax(logits, dim=0)
        raw_weights = torch.exp(logits)
        if not torch.allclose(normalized_weights, raw_weights, rtol=1e-2, atol=1e-2):
            print(
                "Procedure result is outside feasible region. Mu values are not "
                "normalized. Returning normalized values."
            )
        return normalized_weights

    @property
    def amplitudes(self):
        return torch.exp(self.log_amplitudes)

    def _additional_regularization(self):
        _, orth = self.projector(self.m)
        residual = orth - self.projector.mean
        return self.subspace_reg_lambda * (residual**2).sum()

    def parameter_blocks(self):
        blocks = {
            "weights": [self.log_amplitudes],
            "means": [self.m],
            "scales": [self.log_sigma],
        }
        blocks["projector"] = (
            list(self.projector.parameters()) if self.train_projector else []
        )
        blocks["geometry"] = blocks["means"] + blocks["projector"]
        blocks["component_params"] = blocks["means"] + blocks["scales"]
        blocks["all"] = (
            blocks["weights"] + blocks["component_params"] + blocks["projector"]
        )
        return blocks

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

    def fitted_isotropic_variances(self):
        d_minus_r = self.d - self.projection_dim
        orthogonal_noise = torch.as_tensor(
            self.orthogonal_noise_scale,
            dtype=self.sigma.dtype,
            device=self.sigma.device,
        )
        return (
            self.projection_dim * self.sigma.square()
            + d_minus_r * orthogonal_noise.square()
        ) / self.d

    def fitted_covariances(self):
        projection = self.projector.projector_matrix()
        d = projection.shape[0]
        orthogonal_projection = (
            torch.eye(d, dtype=projection.dtype, device=projection.device) - projection
        )
        orthogonal_noise = torch.as_tensor(
            self.orthogonal_noise_scale,
            dtype=projection.dtype,
            device=projection.device,
        )
        return (
            self.sigma.square()[:, None, None] * projection[None, :, :]
            + orthogonal_noise.square() * orthogonal_projection[None, :, :]
        )
