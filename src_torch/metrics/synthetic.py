from __future__ import annotations

from itertools import permutations
from typing import Dict

import torch
from scipy.optimize import linear_sum_assignment

from src.metrics.common import log_prob_gmm, prepare_gmm_tensors, sample_gmm


def evaluate_synthetic_gmm(
    est_means: torch.Tensor,
    est_covariances: torch.Tensor,
    est_weights: torch.Tensor,
    true_means: torch.Tensor,
    true_covariances: torch.Tensor,
    true_weights: torch.Tensor,
    *,
    num_samples_ll: int = 50_000,
    center_recovery_relative_tol: float = 0.1,
    exact_matching_max_components: int = 5,
    eps: float = 1e-9,
) -> Dict[str, float]:
    """Evaluate a fitted GMM against synthetic ground truth.

    The default metrics intentionally avoid covariance-aware component matching:
    likelihood uses the full distributions, while center recovery only compares
    the fitted and true point clouds of means.
    """

    with torch.no_grad():
        true_means = torch.as_tensor(true_means, dtype=torch.float64)
        true_covariances = torch.as_tensor(true_covariances, dtype=torch.float64)
        true_weights = torch.as_tensor(true_weights, dtype=torch.float64)

        est_means = torch.as_tensor(
            est_means, dtype=true_means.dtype, device=true_means.device
        )
        est_covariances = torch.as_tensor(
            est_covariances,
            dtype=true_means.dtype,
            device=true_means.device,
        )
        est_weights = torch.as_tensor(
            est_weights, dtype=true_means.dtype, device=true_means.device
        )

        est_means, est_covariances, est_weights = prepare_gmm_tensors(
            est_means,
            est_covariances,
            est_weights,
            device=true_means.device,
            dtype=true_means.dtype,
            eps=eps,
        )
        true_means, true_covariances, true_weights = prepare_gmm_tensors(
            true_means,
            true_covariances,
            true_weights,
            device=true_means.device,
            dtype=true_means.dtype,
            eps=eps,
        )

        x_test = sample_gmm(
            true_means,
            true_covariances,
            true_weights,
            num_samples_ll,
            eps=eps,
        )
        test_log_likelihood = log_prob_gmm(
            x_test,
            est_means,
            est_covariances,
            est_weights,
            eps=eps,
        ).mean()
        means_distance, center_recovery_rate = center_matching_metrics(
            true_means,
            est_means,
            center_recovery_relative_tol=center_recovery_relative_tol,
            exact_matching_max_components=exact_matching_max_components,
            eps=eps,
        )

        return {
            "test_log_likelihood": float(test_log_likelihood.item()),
            "means_distance": float(means_distance),
            "center_recovery_rate": float(center_recovery_rate),
        }


def center_matching_metrics(
    true_means: torch.Tensor,
    est_means: torch.Tensor,
    *,
    center_recovery_relative_tol: float = 0.1,
    exact_matching_max_components: int = 5,
    eps: float = 1e-12,
) -> tuple[float, float]:
    true_means = torch.as_tensor(true_means, dtype=torch.float64)
    est_means = torch.as_tensor(
        est_means, dtype=true_means.dtype, device=true_means.device
    )

    if true_means.ndim != 2:
        raise ValueError(
            f"true_means must have shape [K, D], got {tuple(true_means.shape)}"
        )
    if est_means.ndim != 2:
        raise ValueError(
            f"est_means must have shape [K, D], got {tuple(est_means.shape)}"
        )
    if true_means.shape[1] != est_means.shape[1]:
        raise ValueError(
            f"Mean dimensions must match, got {true_means.shape[1]} and {est_means.shape[1]}"
        )
    if true_means.shape[0] == 0:
        raise ValueError("true_means must contain at least one center")

    scale = mean_pairwise_center_distance(true_means, eps=eps)
    if est_means.shape[0] == 0:
        distances = torch.full(
            (true_means.shape[0],),
            scale,
            dtype=true_means.dtype,
            device=true_means.device,
        )
    else:
        pairwise = torch.cdist(true_means, est_means, p=2)
        distances = match_center_distances(
            pairwise,
            unmatched_cost=scale,
            exact_matching_max_components=exact_matching_max_components,
        )

    means_distance = distances.mean()
    threshold = center_recovery_relative_tol * scale
    center_recovery_rate = (distances <= threshold).to(torch.float64).mean()
    return float(means_distance.item()), float(center_recovery_rate.item())


def match_center_distances(
    pairwise_distances: torch.Tensor,
    *,
    unmatched_cost: float | torch.Tensor,
    exact_matching_max_components: int = 5,
) -> torch.Tensor:
    if pairwise_distances.ndim != 2:
        raise ValueError(
            f"pairwise_distances must have shape [K_true, K_est], got {tuple(pairwise_distances.shape)}"
        )

    k_true, k_est = pairwise_distances.shape
    unmatched = torch.as_tensor(
        unmatched_cost,
        dtype=pairwise_distances.dtype,
        device=pairwise_distances.device,
    )
    if k_true == 0:
        return torch.empty(
            0, dtype=pairwise_distances.dtype, device=pairwise_distances.device
        )
    if k_est == 0:
        return torch.full(
            (k_true,),
            float(unmatched.item()),
            dtype=pairwise_distances.dtype,
            device=pairwise_distances.device,
        )

    max_components = max(k_true, k_est)
    if max_components <= exact_matching_max_components:
        return _exact_match_distances(pairwise_distances, unmatched)
    return _hungarian_match_distances(pairwise_distances, unmatched)


def mean_pairwise_center_distance(
    true_means: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    k = true_means.shape[0]
    if k < 2:
        return true_means.new_tensor(1.0)

    distances = torch.cdist(true_means, true_means, p=2)
    mask = ~torch.eye(k, dtype=torch.bool, device=true_means.device)
    mean_distance = distances[mask].mean()
    return mean_distance.clamp_min(eps)


def _exact_match_distances(
    pairwise_distances: torch.Tensor, unmatched_cost: torch.Tensor
) -> torch.Tensor:
    k_true, k_est = pairwise_distances.shape
    if k_est >= k_true:
        best: torch.Tensor | None = None
        for cols in permutations(range(k_est), k_true):
            candidate = pairwise_distances[
                torch.arange(k_true, device=pairwise_distances.device), list(cols)
            ]
            if best is None or candidate.sum() < best.sum():
                best = candidate
        if best is None:
            raise RuntimeError("Failed to construct exact center matching")
        return best

    best: torch.Tensor | None = None
    for rows in permutations(range(k_true), k_est):
        candidate = torch.full(
            (k_true,),
            float(unmatched_cost.item()),
            dtype=pairwise_distances.dtype,
            device=pairwise_distances.device,
        )
        candidate[list(rows)] = pairwise_distances[
            list(rows), torch.arange(k_est, device=pairwise_distances.device)
        ]
        if best is None or candidate.sum() < best.sum():
            best = candidate
    if best is None:
        raise RuntimeError("Failed to construct exact center matching")
    return best


def _hungarian_match_distances(
    pairwise_distances: torch.Tensor, unmatched_cost: torch.Tensor
) -> torch.Tensor:
    k_true, k_est = pairwise_distances.shape
    n_cols = max(k_true, k_est)
    cost = torch.full(
        (k_true, n_cols),
        float(unmatched_cost.item()),
        dtype=pairwise_distances.dtype,
        device=pairwise_distances.device,
    )
    cost[:, :k_est] = pairwise_distances
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

    matched = torch.full(
        (k_true,),
        float(unmatched_cost.item()),
        dtype=pairwise_distances.dtype,
        device=pairwise_distances.device,
    )
    for row, col in zip(row_ind, col_ind):
        if col < k_est:
            matched[row] = pairwise_distances[row, col]
    return matched
