from __future__ import annotations

from typing import Optional

import torch


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum().clamp_min(1e-12)


def isotropic_covariances(sigmas: torch.Tensor, n_features: int) -> torch.Tensor:
    if sigmas.ndim != 1:
        raise ValueError(f"sigmas must have shape [K], got {tuple(sigmas.shape)}")

    eye = torch.eye(n_features, dtype=sigmas.dtype, device=sigmas.device)
    return sigmas.square()[:, None, None] * eye[None, :, :]


def trace_equivalent_sigmas(
    covariances: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    if covariances.ndim != 3:
        raise ValueError(
            f"covariances must have shape [K, D, D], got {tuple(covariances.shape)}"
        )

    variances = torch.diagonal(covariances, dim1=1, dim2=2).mean(dim=1)
    return torch.sqrt(variances.clamp_min(eps))


def validate_gmm_tensors(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    name: str,
) -> None:
    if means.ndim != 2:
        raise ValueError(
            f"{name}_means must have shape [K, D], got {tuple(means.shape)}"
        )
    if covariances.ndim != 3:
        raise ValueError(
            f"{name}_covariances must have shape [K, D, D], got {tuple(covariances.shape)}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"{name}_weights must have shape [K], got {tuple(weights.shape)}"
        )

    k, d = means.shape
    if covariances.shape != (k, d, d):
        raise ValueError(
            f"{name}_covariances must have shape {(k, d, d)}, got {tuple(covariances.shape)}"
        )
    if weights.shape[0] != k:
        raise ValueError(
            f"{name}_weights must have length K={k}, got {weights.shape[0]}"
        )
    if torch.any(weights < 0):
        raise ValueError(f"{name}_weights must be nonnegative")
    if float(weights.sum()) <= 0:
        raise ValueError(f"{name}_weights must sum to a positive value")


def prepare_gmm_tensors(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = means.device if device is None else device
    dtype = means.dtype if dtype is None else dtype

    means = means.to(device=device, dtype=dtype)
    covariances = covariances.to(device=device, dtype=dtype)
    weights = normalize_weights(weights.to(device=device, dtype=dtype))
    covariances = regularize_covariances(covariances, eps=eps)
    return means, covariances, weights


def regularize_covariances(
    covariances: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    d = covariances.shape[-1]
    eye = torch.eye(d, dtype=covariances.dtype, device=covariances.device)
    return covariances + eps * eye[None, :, :]


def cholesky_with_jitter(
    covariances: torch.Tensor,
    *,
    initial_jitter: float = 1e-9,
    max_tries: int = 6,
) -> torch.Tensor:
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    d = covariances.shape[-1]
    eye = torch.eye(d, dtype=covariances.dtype, device=covariances.device)

    jitter = initial_jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(covariances + jitter * eye[None, :, :])
        except RuntimeError:
            jitter *= 10.0

    return torch.linalg.cholesky(covariances + jitter * eye[None, :, :])


def sample_gmm(
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    validate_gmm_tensors(means, covariances, weights, "gmm")
    means, covariances, weights = prepare_gmm_tensors(
        means,
        covariances,
        weights,
        device=means.device,
        dtype=means.dtype,
        eps=eps,
    )

    _, d = means.shape
    component_ids = torch.multinomial(weights, num_samples=n_samples, replacement=True)
    chol = cholesky_with_jitter(covariances, initial_jitter=eps)
    noise = torch.randn(n_samples, d, dtype=means.dtype, device=means.device)
    transformed = torch.bmm(chol[component_ids], noise.unsqueeze(-1)).squeeze(-1)
    return means[component_ids] + transformed


def log_prob_gmm(
    x: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    validate_gmm_tensors(means, covariances, weights, "gmm")

    x = x.to(device=means.device, dtype=means.dtype)
    means, covariances, weights = prepare_gmm_tensors(
        means,
        covariances,
        weights,
        device=means.device,
        dtype=means.dtype,
        eps=eps,
    )

    n, d = x.shape
    k = means.shape[0]
    diff = x[:, None, :] - means[None, :, :]  # [N, K, D]

    chol = cholesky_with_jitter(covariances, initial_jitter=eps)
    flat_diff = diff.permute(1, 2, 0).reshape(k, d, n)
    solved = torch.cholesky_solve(flat_diff, chol).reshape(k, d, n).permute(2, 0, 1)
    mahalanobis = (diff * solved).sum(dim=-1)

    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
    log_norm = 0.5 * (d * log_2pi + log_det)
    log_weights = torch.log(weights.clamp_min(eps))

    component_log_probs = -0.5 * mahalanobis - log_norm[None, :] + log_weights[None, :]
    return torch.logsumexp(component_log_probs, dim=1)


def estimate_js_divergence(
    p_means: torch.Tensor,
    p_covariances: torch.Tensor,
    p_weights: torch.Tensor,
    q_means: torch.Tensor,
    q_covariances: torch.Tensor,
    q_weights: torch.Tensor,
    n_samples: int,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    x_p = sample_gmm(p_means, p_covariances, p_weights, n_samples, eps=eps)
    x_q = sample_gmm(q_means, q_covariances, q_weights, n_samples, eps=eps)

    log_p_xp = log_prob_gmm(x_p, p_means, p_covariances, p_weights, eps=eps)
    log_q_xp = log_prob_gmm(x_p, q_means, q_covariances, q_weights, eps=eps)
    log_q_xq = log_prob_gmm(x_q, q_means, q_covariances, q_weights, eps=eps)
    log_p_xq = log_prob_gmm(x_q, p_means, p_covariances, p_weights, eps=eps)

    log_half = torch.log(torch.tensor(0.5, dtype=x_p.dtype, device=x_p.device))
    log_m_xp = torch.logsumexp(
        torch.stack([log_half + log_p_xp, log_half + log_q_xp]), dim=0
    )
    log_m_xq = torch.logsumexp(
        torch.stack([log_half + log_p_xq, log_half + log_q_xq]), dim=0
    )
    return 0.5 * (log_p_xp - log_m_xp).mean() + 0.5 * (log_q_xq - log_m_xq).mean()


def subsample_rows(x: torch.Tensor, n_max: int) -> torch.Tensor:
    if x.shape[0] <= n_max:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n_max]
    return x[idx]


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).T
    return (x_norm + y_norm - 2.0 * (x @ y.T)).clamp_min(0.0)


def mmd_rbf(
    x: torch.Tensor, y: torch.Tensor, bandwidth: Optional[float], eps: float = 1e-12
) -> torch.Tensor:
    if bandwidth is None:
        z = subsample_rows(torch.cat([x, y], dim=0), min(2048, x.shape[0] + y.shape[0]))
        sq_dists = pairwise_sq_dists(z, z)
        mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        values = sq_dists[mask]
        values = values[values > 0]
        bandwidth = (
            1.0
            if values.numel() == 0
            else max(torch.sqrt(values.median() + eps).item(), 1e-6)
        )

    h = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
    return (
        torch.exp(-pairwise_sq_dists(x, x) / (2.0 * h * h)).mean()
        + torch.exp(-pairwise_sq_dists(y, y) / (2.0 * h * h)).mean()
        - 2.0 * torch.exp(-pairwise_sq_dists(x, y) / (2.0 * h * h)).mean()
    )


def energy_distance(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    def pairwise_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(pairwise_sq_dists(a, b) + eps)

    def mean_offdiag(a: torch.Tensor) -> torch.Tensor:
        if a.shape[0] < 2:
            return torch.zeros((), dtype=a.dtype, device=a.device)
        mask = ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
        return a[mask].mean()

    d_xy = pairwise_dists(x, y).mean()
    d_xx = mean_offdiag(pairwise_dists(x, x))
    d_yy = mean_offdiag(pairwise_dists(y, y))
    return 2.0 * d_xy - d_xx - d_yy
