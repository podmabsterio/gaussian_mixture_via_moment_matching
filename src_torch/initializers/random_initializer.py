from src_torch.initializers.base_initializer import BaseInitializer

import torch


class RandomUniformInitializer(BaseInitializer):
    def __init__(
        self,
        n_components,
        m_min,
        m_max,
        sigma_min=0.1,
        sigma_max=1.0,
        mu_min=0,
        mu_max=1,
    ):
        super().__init__(n_components)
        self.m_min = m_min
        self.m_max = m_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mu_min = mu_min
        self.mu_max = mu_max

    def get_initial_params(self, X):
        dim = X.shape[1]
        m = (
            torch.rand(self.n_components, dim, device=X.device, dtype=X.dtype)
            * (self.m_max - self.m_min)
            + self.m_min
        )
        sigma = (
            torch.rand(self.n_components, device=X.device, dtype=X.dtype)
            * (self.sigma_max - self.sigma_min)
            + self.sigma_min
        )
        mu = (
            torch.rand(self.n_components, device=X.device, dtype=X.dtype)
            * (self.mu_max - self.mu_min)
            + self.mu_min
        )
        mu = mu / mu.sum().clamp_min(1e-12)  # Normalize to sum to 1
        return {"init_m": m, "init_sigma": sigma, "init_mu": mu}
