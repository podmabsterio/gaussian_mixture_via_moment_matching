from __future__ import annotations

from typing import Dict

import torch
from scipy.optimize import linear_sum_assignment

from src_torch.metrics.common import (
    estimate_js_divergence,
    isotropic_covariances,
    log_prob_gmm,
    prepare_gmm_tensors,
    sample_gmm,
    trace_equivalent_sigmas,
    validate_gmm_tensors,
)


def evaluate_isotropic_synthetic_gmm(
    est_means: torch.Tensor,
    est_sigmas: torch.Tensor,
    est_weights: torch.Tensor,
    true_means: torch.Tensor,
    true_sigmas: torch.Tensor,
    true_weights: torch.Tensor,
    *,
    num_samples_ll: int = 50_000,
    num_samples_js: int = 50_000,
    weight_cost: float = 1.0,
    center_tol: float = 1.0,
    sigma_rel_tol: float = 0.25,
    weight_tol: float = 0.05,
    eps: float = 1e-9,
) -> Dict[str, float]:
    with torch.no_grad():
        n_features = true_means.shape[1]
        est_covariances = isotropic_covariances(est_sigmas, n_features)
        true_covariances = isotropic_covariances(true_sigmas, n_features)

        device = true_means.device
        dtype = true_means.dtype
        est_means, est_covariances, est_weights = prepare_gmm_tensors(
            est_means, est_covariances, est_weights, device=device, dtype=dtype, eps=eps
        )
        true_means, true_covariances, true_weights = prepare_gmm_tensors(
            true_means,
            true_covariances,
            true_weights,
            device=device,
            dtype=dtype,
            eps=eps,
        )
        est_sigmas = est_sigmas.to(device=device, dtype=dtype)
        true_sigmas = true_sigmas.to(device=device, dtype=dtype)

        x_test = sample_gmm(
            true_means, true_covariances, true_weights, num_samples_ll, eps=eps
        )
        test_log_likelihood = log_prob_gmm(
            x_test, est_means, est_covariances, est_weights, eps=eps
        ).mean()
        js_divergence = estimate_js_divergence(
            true_means,
            true_covariances,
            true_weights,
            est_means,
            est_covariances,
            est_weights,
            num_samples_js,
            eps=eps,
        )
        parametric_distance, recovery_rate = (
            _hungarian_isotropic_parametric_distance_and_crr(
                est_means,
                est_sigmas,
                est_weights,
                true_means,
                true_sigmas,
                true_weights,
                weight_cost=weight_cost,
                center_tol=center_tol,
                sigma_rel_tol=sigma_rel_tol,
                weight_tol=weight_tol,
                eps=eps,
            )
        )

        return {
            "js_divergence": float(js_divergence.item()),
            "test_log_likelihood": float(test_log_likelihood.item()),
            "parametric_distance": float(parametric_distance),
            "component_recovery_rate": float(recovery_rate),
        }


def evaluate_anisotropic_synthetic_gmm(
    est_means: torch.Tensor,
    est_covariances: torch.Tensor,
    est_weights: torch.Tensor,
    true_means: torch.Tensor,
    true_covariances: torch.Tensor,
    true_weights: torch.Tensor,
    *,
    num_samples_ll: int = 50_000,
    num_samples_js: int = 50_000,
    weight_cost: float = 1.0,
    center_tol: float = 1.0,
    covariance_rel_tol: float | None = None,
    sigma_rel_tol: float | None = None,
    weight_tol: float = 0.05,
    covariance_cost: str = "w2",
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Evaluate synthetic GMMs with arbitrary full covariance matrices."""
    if covariance_rel_tol is None:
        covariance_rel_tol = 0.25 if sigma_rel_tol is None else sigma_rel_tol

    with torch.no_grad():
        validate_gmm_tensors(est_means, est_covariances, est_weights, "est")
        validate_gmm_tensors(true_means, true_covariances, true_weights, "true")

        device = true_means.device
        dtype = true_means.dtype
        est_means, est_covariances, est_weights = prepare_gmm_tensors(
            est_means, est_covariances, est_weights, device=device, dtype=dtype, eps=eps
        )
        true_means, true_covariances, true_weights = prepare_gmm_tensors(
            true_means,
            true_covariances,
            true_weights,
            device=device,
            dtype=dtype,
            eps=eps,
        )

        x_test = sample_gmm(
            true_means, true_covariances, true_weights, num_samples_ll, eps=eps
        )
        test_log_likelihood = log_prob_gmm(
            x_test, est_means, est_covariances, est_weights, eps=eps
        ).mean()

        js_divergence = estimate_js_divergence(
            true_means,
            true_covariances,
            true_weights,
            est_means,
            est_covariances,
            est_weights,
            num_samples_js,
            eps=eps,
        )

        parametric_distance, recovery_rate = _hungarian_parametric_distance_and_crr(
            est_means,
            est_covariances,
            est_weights,
            true_means,
            true_covariances,
            true_weights,
            weight_cost=weight_cost,
            center_tol=center_tol,
            covariance_rel_tol=covariance_rel_tol,
            weight_tol=weight_tol,
            covariance_cost=covariance_cost,
            eps=eps,
        )

        return {
            "js_divergence": float(js_divergence.item()),
            "test_log_likelihood": float(test_log_likelihood.item()),
            "parametric_distance": float(parametric_distance),
            "component_recovery_rate": float(recovery_rate),
        }


def _hungarian_parametric_distance_and_crr(
    est_means: torch.Tensor,
    est_covariances: torch.Tensor,
    est_weights: torch.Tensor,
    true_means: torch.Tensor,
    true_covariances: torch.Tensor,
    true_weights: torch.Tensor,
    *,
    weight_cost: float,
    center_tol: float,
    covariance_rel_tol: float,
    weight_tol: float,
    covariance_cost: str,
    eps: float,
) -> tuple[float, float]:
    k_true, d = true_means.shape
    k_est = est_means.shape[0]
    m = max(k_true, k_est)

    center_cost = ((true_means[:, None, :] - est_means[None, :, :]) ** 2).sum(dim=-1)
    cov_cost = _pairwise_covariance_cost(
        true_covariances,
        est_covariances,
        mode=covariance_cost,
        eps=eps,
    )
    weight_mismatch = weight_cost * (true_weights[:, None] - est_weights[None, :]).pow(
        2
    )
    pair_cost = center_cost + cov_cost + weight_mismatch

    if k_true != k_est:
        max_pair_cost = (
            float(pair_cost.max().detach().cpu()) if pair_cost.numel() else 1.0
        )
        dummy_cost = max_pair_cost + 1e6
        cost_matrix = torch.full(
            (m, m), dummy_cost, dtype=true_means.dtype, device=true_means.device
        )
        cost_matrix[:k_true, :k_est] = pair_cost
    else:
        cost_matrix = pair_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    true_match_cost = torch.empty(
        k_true, dtype=true_means.dtype, device=true_means.device
    )
    true_match_cost.fill_(float(cost_matrix.max().item()))

    true_sigmas = trace_equivalent_sigmas(true_covariances, eps=eps)
    recovered = 0
    for r, c in zip(row_ind, col_ind):
        if r >= k_true:
            continue
        if c >= k_est:
            true_match_cost[r] = cost_matrix[r, c]
            continue

        true_match_cost[r] = pair_cost[r, c]
        mu_err = torch.norm(true_means[r] - est_means[c], p=2)
        cov_rel_err = torch.linalg.norm(
            true_covariances[r] - est_covariances[c], ord="fro"
        ) / (torch.linalg.norm(true_covariances[r], ord="fro").clamp_min(eps))
        w_err = torch.abs(true_weights[r] - est_weights[c])
        center_threshold = center_tol * true_sigmas[r] * (d**0.5)
        recovered += int(
            (mu_err <= center_threshold)
            and (cov_rel_err <= covariance_rel_tol)
            and (w_err <= weight_tol)
        )

    parametric_distance = float((true_weights * true_match_cost).sum().item())
    recovery_rate = float(recovered / max(k_true, 1))
    return parametric_distance, recovery_rate


def _hungarian_isotropic_parametric_distance_and_crr(
    est_means: torch.Tensor,
    est_sigmas: torch.Tensor,
    est_weights: torch.Tensor,
    true_means: torch.Tensor,
    true_sigmas: torch.Tensor,
    true_weights: torch.Tensor,
    *,
    weight_cost: float,
    center_tol: float,
    sigma_rel_tol: float,
    weight_tol: float,
    eps: float,
) -> tuple[float, float]:
    k_true, d = true_means.shape
    k_est = est_means.shape[0]
    m = max(k_true, k_est)

    center_cost = ((true_means[:, None, :] - est_means[None, :, :]) ** 2).sum(dim=-1)
    sigma_cost = d * (true_sigmas[:, None] - est_sigmas[None, :]).pow(2)
    weight_mismatch = weight_cost * (true_weights[:, None] - est_weights[None, :]).pow(
        2
    )
    pair_cost = center_cost + sigma_cost + weight_mismatch

    if k_true != k_est:
        max_pair_cost = (
            float(pair_cost.max().detach().cpu()) if pair_cost.numel() else 1.0
        )
        dummy_cost = max_pair_cost + 1e6
        cost_matrix = torch.full(
            (m, m), dummy_cost, dtype=true_means.dtype, device=true_means.device
        )
        cost_matrix[:k_true, :k_est] = pair_cost
    else:
        cost_matrix = pair_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    true_match_cost = torch.empty(
        k_true, dtype=true_means.dtype, device=true_means.device
    )
    true_match_cost.fill_(float(cost_matrix.max().item()))

    recovered = 0
    for r, c in zip(row_ind, col_ind):
        if r >= k_true:
            continue
        if c >= k_est:
            true_match_cost[r] = cost_matrix[r, c]
            continue

        true_match_cost[r] = pair_cost[r, c]
        mu_err = torch.norm(true_means[r] - est_means[c], p=2)
        sigma_rel_err = torch.abs(true_sigmas[r] - est_sigmas[c]) / true_sigmas[
            r
        ].clamp_min(eps)
        w_err = torch.abs(true_weights[r] - est_weights[c])
        center_threshold = center_tol * true_sigmas[r] * (d**0.5)
        recovered += int(
            (mu_err <= center_threshold)
            and (sigma_rel_err <= sigma_rel_tol)
            and (w_err <= weight_tol)
        )

    return float((true_weights * true_match_cost).sum().item()), float(
        recovered / max(k_true, 1)
    )


def _pairwise_covariance_cost(
    true_covariances: torch.Tensor,
    est_covariances: torch.Tensor,
    *,
    mode: str,
    eps: float,
) -> torch.Tensor:
    if mode == "frobenius":
        diff = true_covariances[:, None, :, :] - est_covariances[None, :, :, :]
        return (diff * diff).sum(dim=(-1, -2))
    if mode != "w2":
        raise ValueError(f"Unknown covariance_cost: {mode}")

    costs = torch.empty(
        true_covariances.shape[0],
        est_covariances.shape[0],
        dtype=true_covariances.dtype,
        device=true_covariances.device,
    )
    for i, cov_true in enumerate(true_covariances):
        for j, cov_est in enumerate(est_covariances):
            costs[i, j] = _gaussian_w2_covariance_cost(cov_true, cov_est, eps=eps)
    return costs


def _gaussian_w2_covariance_cost(
    cov_a: torch.Tensor,
    cov_b: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    sqrt_b = _symmetric_matrix_sqrt(cov_b, eps=eps)
    middle = sqrt_b @ cov_a @ sqrt_b
    sqrt_middle = _symmetric_matrix_sqrt(middle, eps=eps)
    return torch.trace(cov_a + cov_b - 2.0 * sqrt_middle).clamp_min(0.0)


def _symmetric_matrix_sqrt(matrix: torch.Tensor, *, eps: float) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(eps)
    return (eigvecs * torch.sqrt(eigvals)[None, :]) @ eigvecs.T
