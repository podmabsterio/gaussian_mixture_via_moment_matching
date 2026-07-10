from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import numpy as np

Array = np.ndarray


@dataclass(slots=True)
class MixtureSpec:
    """Configuration for one synthetic mixture dataset."""

    n_samples: int
    n_features: int
    n_components: int

    separation: float = 3.0
    variance_spread: float = 0.0
    anisotropy: float = 1.0

    subspace_dim: Optional[int] = None
    subspace_noise: float = 0.0

    weight_mode: str = "uniform"
    weight_concentration: float = 10.0

    base_scale: float = 1.0
    random_seed: Optional[int] = None

    scenario: str = "isotropic"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        """Ambient dimension alias."""
        return self.n_features

    @property
    def k(self) -> int:
        """Number of components alias."""
        return self.n_components

    @property
    def n(self) -> int:
        """Number of samples alias."""
        return self.n_samples


@dataclass(slots=True)
class GeneratedDataset:
    """Synthetic mixture dataset with ground-truth parameters."""

    X: Array
    y: Array
    means: Array
    covariances: Array
    weights: Array
    spec: MixtureSpec

    subspace_basis: Optional[Array] = None
    subspace_origin: Optional[Array] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return dataset fields as a dictionary."""
        return {
            "X": self.X,
            "y": self.y,
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights,
            "subspace_basis": self.subspace_basis,
            "subspace_origin": self.subspace_origin,
            "spec": self.spec,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CenterResult:
    """Generated centers and optional geometry metadata."""

    means: Array
    subspace_basis: Optional[Array] = None
    subspace_origin: Optional[Array] = None
    metadata: dict[str, Any] = field(default_factory=dict)
