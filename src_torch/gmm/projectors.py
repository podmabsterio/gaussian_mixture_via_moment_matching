import torch
import torch.nn as nn

import torch


@torch.no_grad()
def fit_affine_subspace(points: torch.Tensor, k: int):
    if points.ndim != 2:
        raise ValueError(f"Expected points of shape (n, d), got {tuple(points.shape)}")

    n, d = points.shape
    if n < 1:
        raise ValueError("points must contain at least one point")
    if not (0 <= k <= d):
        raise ValueError(f"Expected 0 <= k <= d, got k={k}, d={d}")

    mean = points.mean(dim=0)
    centered = points - mean

    if k == 0:
        Q = points.new_empty(d, 0)
        return mean, Q

    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    Q = Vh[:k].T.contiguous()  # (d, min(k, n, d))

    if Q.shape[1] < k:
        candidates = torch.eye(d, dtype=points.dtype, device=points.device)
        Q, _ = torch.linalg.qr(torch.cat([Q, candidates], dim=1), mode="reduced")
        Q = Q[:, :k].contiguous()

    return mean, Q


class AffineOrthogonalProjectorQR(nn.Module):
    def __init__(
        self,
        d: int,
        k: int,
        *,
        trainable_mean: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()

        if not (0 <= k <= d):
            raise ValueError(f"Expected 0 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k

        if k > 0:
            self.W = nn.Parameter(torch.randn(d, k, dtype=dtype, device=device))
        else:
            self.W = nn.Parameter(torch.empty(d, 0, dtype=dtype, device=device))

        mean0 = torch.zeros(d, dtype=dtype, device=device)
        if trainable_mean:
            self.mean = nn.Parameter(mean0)
        else:
            self.register_buffer("mean", mean0)

    def orthonormal_basis(self) -> torch.Tensor:
        Q, R = torch.linalg.qr(self.W, mode="reduced")
        signs = torch.sign(torch.diag(R))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
        return Q

    @property
    def Q(self) -> torch.Tensor:
        return self.orthonormal_basis()

    def projector_matrix(self) -> torch.Tensor:
        Q = self.Q
        return Q @ Q.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected last dim = {self.d}, got {x.shape[-1]}")

        Q = self.Q
        xc = x - self.mean
        xc_proj = (xc @ Q) @ Q.T
        return self.mean + xc_proj, self.mean + (xc - xc_proj)

    @torch.no_grad()
    def initialize_by_points(self, points: torch.Tensor):
        mean0, Q0 = fit_affine_subspace(points, self.k)

        if mean0.shape != (self.d,):
            raise ValueError(
                f"Point dimension mismatch: expected d={self.d}, got {mean0.shape[0]}"
            )

        self.mean.copy_(mean0)
        self.W.copy_(Q0)

        return self


import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torch.nn.utils import parametrize


class AffineOrthogonalProjectorParam(nn.Module):
    """
    Аффинный ортогональный проектор:
        x -> mean + Q Q^T (x - mean)

    Где Q задается как orthogonal-parametrized weight слоя nn.Linear(k, d, bias=False).
    """

    def __init__(
        self,
        d: int,
        k: int,
        *,
        orthogonal_map: str = "householder",
        use_trivialization: bool = True,
        trainable_mean: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()

        if not (0 <= k <= d):
            raise ValueError(f"Expected 0 <= k <= d, got k={k}, d={d}")

        self.d = d
        self.k = k
        self.orthogonal_map = orthogonal_map
        self.use_trivialization = use_trivialization

        self.basis = nn.Linear(
            in_features=max(
                k, 1
            ),  # nn.Linear не любит 0 in_features в части старых сборок
            out_features=d,
            bias=False,
            dtype=dtype,
            device=device,
        )

        if k == 0:
            self._has_basis = False
        else:
            self._has_basis = True
            orthogonal(
                self.basis,
                name="weight",
                orthogonal_map=orthogonal_map,
                use_trivialization=use_trivialization,
            )

        mean0 = torch.zeros(d, dtype=dtype, device=device)
        if trainable_mean:
            self.mean = nn.Parameter(mean0)
        else:
            self.register_buffer("mean", mean0)

    @property
    def Q(self) -> torch.Tensor:
        if self.k == 0:
            return self.mean.new_empty(self.d, 0)
        return self.basis.weight[:, : self.k]

    def projector_matrix(self) -> torch.Tensor:
        Q = self.Q
        return Q @ Q.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected last dim = {self.d}, got {x.shape[-1]}")

        if self.k == 0:
            return self.mean.expand_as(x), x

        Q = self.Q
        xc = x - self.mean
        xc_proj = (xc @ Q) @ Q.T
        return self.mean + xc_proj, self.mean + (xc - xc_proj)

    def _rebuild_parametrization(self):
        if self.k == 0:
            return

        orthogonal(
            self.basis,
            name="weight",
            orthogonal_map=self.orthogonal_map,
            use_trivialization=self.use_trivialization,
        )

    @torch.no_grad()
    def initialize_by_points(self, points: torch.Tensor):
        mean0, Q0 = fit_affine_subspace(points, self.k)

        if mean0.shape != (self.d,):
            raise ValueError(
                f"Point dimension mismatch: expected d={self.d}, got {mean0.shape[0]}"
            )

        self.mean.copy_(mean0)

        if self.k == 0:
            return self

        if parametrize.is_parametrized(self.basis, "weight"):
            parametrize.remove_parametrizations(
                self.basis,
                "weight",
                leave_parametrized=True,
            )

        self.basis.weight.copy_(Q0)

        self._rebuild_parametrization()

        return self
