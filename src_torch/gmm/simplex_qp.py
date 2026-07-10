from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize


def solve_simplex_least_squares(
    A: torch.Tensor,
    z: torch.Tensor,
    *,
    initial_weights: torch.Tensor | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> torch.Tensor:
    """Solve min 0.5 * ||z - A @ weights||^2 over the probability simplex."""
    if A.ndim != 2:
        raise ValueError(
            f"A must have shape [n_moments, n_components], got {tuple(A.shape)}"
        )
    if z.ndim != 1 or z.shape[0] != A.shape[0]:
        raise ValueError(f"z must have shape [{A.shape[0]}], got {tuple(z.shape)}")
    if A.shape[1] == 0:
        raise ValueError("A must contain at least one component")
    if not torch.isfinite(A).all() or not torch.isfinite(z).all():
        raise ValueError("A and z must contain only finite values")

    component_count = A.shape[1]
    if initial_weights is None:
        initial_weights = torch.full(
            (component_count,),
            1.0 / component_count,
            dtype=A.dtype,
            device=A.device,
        )
    elif initial_weights.shape != (component_count,):
        raise ValueError(
            f"initial_weights must have shape {(component_count,)}, "
            f"got {tuple(initial_weights.shape)}"
        )

    initial_weights = initial_weights.detach().clamp_min(0)
    initial_weights = initial_weights / initial_weights.sum().clamp_min(
        torch.finfo(initial_weights.dtype).eps
    )

    A_np = A.detach().to(device="cpu", dtype=torch.float64).numpy()
    z_np = z.detach().to(device="cpu", dtype=torch.float64).numpy()
    x0 = initial_weights.to(device="cpu", dtype=torch.float64).numpy()

    gram = A_np.T @ A_np
    linear = A_np.T @ z_np

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * weights @ gram @ weights - linear @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return gram @ weights - linear

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * component_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": tolerance, "maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(
            f"Simplex least-squares optimization failed: {result.message}"
        )

    weights = torch.as_tensor(result.x, dtype=A.dtype, device=A.device).clamp_min(0)
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
