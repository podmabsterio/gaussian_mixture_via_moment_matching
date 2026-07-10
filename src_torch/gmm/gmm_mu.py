from src.gmm.gmm import GaussianMomentModel

import torch


class GaussianMomentModelMu(GaussianMomentModel):
    def insert_data(self, X, batch_size=16):
        super().insert_data(X, batch_size=batch_size)
        self._parameters["log_amplitudes"] = self._parameters.pop("logits_mu")

        s_value = self.s[0, 0]
        if self.s.shape[0] != 1 or torch.any(self.s != s_value):
            raise ValueError(
                "GaussianMomentModelMu only supports a single bandwidth value for all centers."
            )

        self.s_single_value = s_value

        with torch.no_grad():
            self.log_amplitudes.add_(self.log_psi_dimension_coef())

    def log_psi_dimension_coef(self):
        log_s = self.s_single_value.log()
        return self.d * (
            log_s - 0.5 * torch.logaddexp(2.0 * self.log_sigma, 2.0 * log_s)
        )

    def psi(self, index_batch=None):
        centers = self.centers
        m = self.m
        sigma = self.sigma
        s = self.s

        q, d = centers.shape
        K = m.shape[0]

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

        out = torch.exp(-dist2 / (2.0 * (sigma2 + s2)))
        out = out.permute(1, 0, 2).reshape(q * B, K)
        return out

    def M(self, index_batch):
        psi_value = self.psi(index_batch)
        return psi_value @ self.amplitudes

    @property
    def mu(self):
        logits = self.log_amplitudes - self.log_psi_dimension_coef()
        # normalized_weights = torch.softmax(logits, dim=0)
        # raw_weights = torch.exp(logits)
        # if not torch.allclose(normalized_weights, raw_weights, rtol=1e-2, atol=1e-2):
        #     print("Procedure result is outside feasible region. Mu values are not normalized. Returning normalized values.")
        return torch.softmax(logits, dim=0)

    @property
    def amplitudes(self):
        return torch.exp(self.log_amplitudes)

    def parameter_blocks(self):
        blocks = {
            "weights": [self.log_amplitudes],
            "means": [self.m],
            "scales": [self.log_sigma],
        }
        blocks["component_params"] = blocks["means"] + blocks["scales"]
        blocks["all"] = blocks["weights"] + blocks["component_params"]
        return blocks
