import numpy as np
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

Array = np.ndarray


@dataclass(slots=True)
class SeparationCalibrator:
    """Calibrates centers to a target Mahalanobis separation."""

    def calibrate(
        self,
        means: Array,
        covariances: Array,
        target_separation: float,
    ) -> Array:
        """Scale centers to match the target separation."""
        current_separation = self.compute_separation(means, covariances)

        if current_separation == 0.0:
            raise ValueError("Cannot calibrate separation for identical centers.")

        center = means.mean(axis=0, keepdims=True)
        scale = target_separation / current_separation

        return center + scale * (means - center)

    def compute_separation(
        self,
        means: Array,
        covariances: Array,
    ) -> float:
        """Compute minimal pairwise Mahalanobis separation."""
        n_components = means.shape[0]
        min_distance = np.inf

        for i in range(n_components):
            for j in range(i + 1, n_components):
                distance = self._pairwise_separation(
                    means[i],
                    means[j],
                    covariances[i],
                    covariances[j],
                )
                min_distance = min(min_distance, distance)

        return float(min_distance)

    def _pairwise_separation(
        self,
        mean_i: Array,
        mean_j: Array,
        covariance_i: Array,
        covariance_j: Array,
    ) -> float:
        """Compute Mahalanobis separation for two components."""
        diff = mean_i - mean_j
        pooled_covariance = 0.5 * (covariance_i + covariance_j)

        solved = np.linalg.solve(pooled_covariance, diff)
        squared_distance = diff @ solved

        return float(np.sqrt(squared_distance))
