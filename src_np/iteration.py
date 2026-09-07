"""Shared progress contract for iterative mixture estimators.

The callback is deliberately independent of the web UI.  Optimizers report a
canonical, immutable mixture snapshot; experiment runners decide whether to
log it, stream it, evaluate metrics, or ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np


def _frozen_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MixtureParameters:
    """Canonical parameters exposed by a mixture optimizer at one iteration."""

    means: np.ndarray
    weights: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        means = _frozen_array(self.means, name="means")
        weights = _frozen_array(self.weights, name="weights").reshape(-1)
        covariances = _frozen_array(self.covariances, name="covariances")
        if means.ndim != 2 or means.shape[0] == 0:
            raise ValueError("means must have shape (n_components, n_features)")
        n_components, n_features = means.shape
        if weights.shape != (n_components,):
            raise ValueError("weights must have one value per component")
        if covariances.shape != (n_components, n_features, n_features):
            raise ValueError(
                "covariances must have shape " "(n_components, n_features, n_features)"
            )
        if np.any(weights < 0) or float(np.sum(weights)) <= 0:
            raise ValueError("weights must be non-negative and have positive sum")
        weights = weights / np.sum(weights)
        weights.setflags(write=False)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "covariances", covariances)

    @classmethod
    def spherical(
        cls,
        means: Any,
        weights: Any,
        sigmas: Any,
    ) -> "MixtureParameters":
        means_array = np.asarray(means, dtype=float)
        sigma_array = np.asarray(sigmas, dtype=float).reshape(-1)
        if means_array.ndim != 2 or sigma_array.shape != (means_array.shape[0],):
            raise ValueError("sigmas must have one value per component")
        identity = np.eye(means_array.shape[1], dtype=float)
        covariances = np.square(sigma_array)[:, None, None] * identity
        return cls(means_array, weights, covariances)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means.copy(),
            "weights": self.weights.copy(),
            "covariances": self.covariances.copy(),
        }


@dataclass(frozen=True)
class IterationSnapshot:
    """State delivered to an optional callback after a meaningful iteration."""

    iteration: int
    loss: float
    parameters: MixtureParameters
    phase: str = "optimization"
    losses: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        iteration = int(self.iteration)
        loss = float(self.loss)
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if not np.isfinite(loss):
            raise ValueError("loss must be finite")
        normalized_losses = {key: float(value) for key, value in self.losses.items()}
        if not all(np.isfinite(value) for value in normalized_losses.values()):
            raise ValueError("loss components must be finite")
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "losses", normalized_losses)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def with_context(
        self,
        *,
        iteration: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> "IterationSnapshot":
        return replace(
            self,
            iteration=iteration,
            metadata={**self.metadata, **(metadata or {})},
        )


IterationCallback = Callable[[IterationSnapshot], None]


__all__ = ["IterationCallback", "IterationSnapshot", "MixtureParameters"]
