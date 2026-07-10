import torch
import torch.nn as nn

from src.gmm.test_functions_utils.centers_generators import (
    BaseCentersGenerator,
    DataCentersGenerator,
)
from src.initializers.base_initializer import BaseInitializer
from src.gmm.test_functions_utils.bandwidth_generators import (
    BaseBandwidthGenerator,
    DimensionAwareBandwidthGenerator,
)


EPS = 1e-8


class GaussianMomentModel(nn.Module):
    def __init__(
        self,
        centers_generator: BaseCentersGenerator = None,
        bandwidth_generator: BaseBandwidthGenerator = None,
        initializer: BaseInitializer = None,
        n_components=3,
        alternating=False,
        entropy_pen_lambda=None,
        precompute_Z=True,
    ):
        super().__init__()
        self.centers_generator = centers_generator
        if self.centers_generator is None:
            self.centers_generator = DataCentersGenerator()
        self.bandwidth_generator = bandwidth_generator
        if self.bandwidth_generator is None:
            self.bandwidth_generator = DimensionAwareBandwidthGenerator(4, 4.0, 0.25)
        self.initializer = initializer
        self.alternating = alternating
        self.entropy_pen_lambda = entropy_pen_lambda
        self.K = n_components
        self.precompute_Z = precompute_Z

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
        self.register_buffer("centers", centers)
        self.register_buffer("s", s)
        self._initialize_test_functions(X)

        if self.precompute_Z:
            Z = self.compute_Z(X, batch_size=batch_size)
            self.register_buffer("X", None)
        else:
            self.register_buffer("X", X)
            Z = None

        self.register_buffer("Z", Z)
        self.register_buffer("z", Z.clone() if Z is not None else None)

        self.logits_mu = nn.Parameter(torch.log(init_mu.clamp_min(1e-12)).clone())
        self.log_sigma = nn.Parameter(torch.log(init_sigma.clone()))
        self.m = nn.Parameter(init_m.clone())

    def _initialize_test_functions(self, X):
        pass

    def compute_Z(self, X, batch_size=None):
        n = X.shape[0]
        if batch_size is None:
            return self.compute_Z_batch(X) / n

        Z = None
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size]
            Z_batch = self.compute_Z_batch(X_batch)
            if Z is None:
                Z = Z_batch
            else:
                Z += Z_batch

        return Z / n

    def num_index_functions(self):
        return self.q

    def compute_Z_batch(self, X, index_batch=None):
        centers = self.centers
        q, d = self.centers.shape
        s = self.s

        B, q_s = s.shape
        assert q_s == q

        if index_batch is not None:
            centers = centers[index_batch, :]
            s = s[:, index_batch]
            q = len(index_batch)

        x2 = (X * X).sum(dim=1, keepdim=True)  # [N, 1]
        c2 = (centers * centers).sum(dim=1).unsqueeze(0)  # [1, q]
        dist2 = x2 + c2 - 2.0 * (X @ centers.T)  # [N, q]
        dist2.clamp_min_(0.0)

        if B == 1:
            Z = torch.exp(-dist2 / (2.0 * s.square()[0])).sum(dim=0)  # [q]
            return Z

        s2 = s.square()[None, :, :]  # [1, B, q]
        Z = torch.exp(-dist2[:, None, :] / (2.0 * s2)).sum(dim=0)  # [B, q]
        return Z.permute(1, 0)  # .reshape(q * B)

    def get_Z(self, index_batch):
        B, q = self.s.shape
        if self.precompute_Z and index_batch is not None:
            q = len(index_batch)
            return self.Z[index_batch].reshape(q * B)
        elif self.precompute_Z:
            return self.Z.reshape(q * B)

        X = self.X
        n = X.shape[0]
        return self.compute_Z_batch(X, index_batch) / n

    def psi(self, index_batch=None):
        centers = self.centers
        m = self.m
        sigma = self.sigma
        s = self.s

        q, d = centers.shape
        K = m.shape[0]

        # if s.ndim == 0:
        #     s = s.expand(1, q)
        # elif s.ndim == 1:
        #     s = s[None, :]

        B, q_s = s.shape
        assert q_s == q

        if index_batch is not None:
            centers = centers[index_batch, :]
            s = s[:, index_batch]
            q = len(index_batch)

        assert m.shape == (K, d)
        assert sigma.shape == (K,)

        c2 = (centers * centers).sum(dim=1, keepdim=True)
        m2 = (m * m).sum(dim=1).unsqueeze(0)
        dist2 = c2 + m2 - 2.0 * (centers @ m.T)
        dist2.clamp_min_(0.0)
        sigma2 = sigma.square()[None, None, :]
        s2 = s.square()[:, :, None]
        dist2 = dist2[None, :, :]

        out = (1.0 + sigma2 / s2).pow(-d / 2.0) * torch.exp(
            -dist2 / (2.0 * (sigma2 + s2))
        )
        out = out.permute(1, 0, 2).reshape(q * B, K)
        return out

    @property
    def mu(self):
        return torch.softmax(self.logits_mu, dim=0)

    def parameter_blocks(self):
        blocks = {
            "weights": [self.logits_mu],
            "means": [self.m],
            "scales": [self.log_sigma],
        }
        blocks["component_params"] = blocks["means"] + blocks["scales"]
        blocks["all"] = blocks["weights"] + blocks["component_params"]
        return blocks

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def M(self, index_batch):
        psi_value = self.psi(index_batch)
        return psi_value @ self.mu

    def _entropy_term(self):
        if self.entropy_pen_lambda is None:
            return 0.0

        mu = self.mu
        return -self.entropy_pen_lambda * torch.sum(mu * mu.log())

    def _additional_regularization(self):
        return 0.0

    def _data_term(self, M, Z):
        if self.alternating:
            z = self.get_z(M, Z)
            data_term_1 = 0.5 * ((Z - z) ** 2).sum()
            data_term_2 = 0.5 * ((z - M) ** 2).sum()
            return data_term_1 + data_term_2

        return 0.5 * ((Z - M) ** 2).sum()

    def get_z(self, M, Z):
        return 0.5 * (Z + M)

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
        return coef * (data_term + entropy_reg + additional_reg)

    # @torch.no_grad()
    # def update_z(self):
    #     self.z = (0.5 * (self.Z + self.M())).clone()

    def sample(self, n_samples):
        with torch.no_grad():
            weights = self.mu
            component_indices = torch.multinomial(
                weights, num_samples=n_samples, replacement=True
            )
            selected_means = self.m[component_indices]
            selected_sigmas = self.sigma[component_indices]

            samples = selected_means + selected_sigmas[:, None] * torch.randn_like(
                selected_means
            )
            return samples
