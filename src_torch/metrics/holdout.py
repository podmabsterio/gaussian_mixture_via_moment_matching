from __future__ import annotations

from typing import Dict, Optional

import torch

from src_torch.metrics.common import (
    energy_distance,
    log_prob_gmm,
    mmd_rbf,
    prepare_gmm_tensors,
    sample_gmm,
    subsample_rows,
    validate_gmm_tensors,
)


def evaluate_gmm_on_holdout(
    est_means: torch.Tensor,
    est_covariances: torch.Tensor,
    est_weights: torch.Tensor,
    holdout_data: torch.Tensor,
    *,
    num_model_samples: int = 50_000,
    max_metric_samples: int = 10_000,
    mmd_bandwidth: Optional[float] = None,
    eps: float = 1e-9,
    **unused_kwargs,
) -> Dict[str, float]:
    """Evaluate a fitted GMM against held-out real data."""

    with torch.no_grad():
        validate_gmm_tensors(est_means, est_covariances, est_weights, "est")
        if holdout_data.ndim != 2:
            raise ValueError(
                f"holdout_data must have shape [N, D], got {tuple(holdout_data.shape)}"
            )
        if holdout_data.shape[0] == 0:
            raise ValueError("holdout_data must contain at least one sample")
        if holdout_data.shape[1] != est_means.shape[1]:
            raise ValueError(
                f"holdout_data must have dimension D={est_means.shape[1]}, got D={holdout_data.shape[1]}"
            )

        est_means, est_covariances, est_weights = prepare_gmm_tensors(
            est_means,
            est_covariances,
            est_weights,
            device=est_means.device,
            dtype=est_means.dtype,
            eps=eps,
        )
        holdout_data = holdout_data.to(device=est_means.device, dtype=est_means.dtype)

        test_log_likelihood = log_prob_gmm(
            holdout_data,
            est_means,
            est_covariances,
            est_weights,
            eps=eps,
        ).mean()

        model_samples = sample_gmm(
            est_means,
            est_covariances,
            est_weights,
            num_model_samples,
            eps=eps,
        )
        x = subsample_rows(holdout_data, max_metric_samples)
        y = subsample_rows(model_samples, max_metric_samples)

        return {
            "test_log_likelihood": float(test_log_likelihood.item()),
            "mmd_rbf": float(mmd_rbf(x, y, mmd_bandwidth, eps=eps).item()),
            "energy_distance": float(energy_distance(x, y, eps=eps).item()),
        }
