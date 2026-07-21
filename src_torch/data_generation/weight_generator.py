from dataclasses import dataclass
from typing import Optional, Sequence
from src_torch.data_generation.dataclass import MixtureSpec

import numpy as np


Array = np.ndarray


class WeightSamplerBase:
    """Base class for mixture weight samplers."""

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> Array:
        """Sample mixture weights."""
        raise NotImplementedError


@dataclass(slots=True)
class EqualWeightSampler(WeightSamplerBase):
    """Samples equal mixture weights."""

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> Array:
        """Return equal weights over components."""
        return np.full(spec.n_components, 1.0 / spec.n_components)


@dataclass(slots=True)
class FixedWeightSampler(WeightSamplerBase):
    """Returns user-provided mixture weights."""

    weights: Sequence[float]

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> Array:
        """Return fixed weights normalized onto the simplex."""
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (spec.n_components,):
            raise ValueError(
                f"FixedWeightSampler expected {spec.n_components} weights, got {weights.shape[0]}."
            )
        if np.any(weights < 0.0):
            raise ValueError("FixedWeightSampler weights must be nonnegative.")

        total = weights.sum()
        if total <= 0.0:
            raise ValueError("FixedWeightSampler weights must sum to a positive value.")

        return weights / total


@dataclass(slots=True)
class DirichletWeightSampler(WeightSamplerBase):
    """Samples weights from a symmetric Dirichlet distribution."""

    concentration: Optional[float] = None

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> Array:
        """Sample random weights on the simplex."""
        concentration = (
            spec.weight_concentration
            if self.concentration is None
            else self.concentration
        )

        alpha = np.full(spec.n_components, concentration)
        return rng.dirichlet(alpha)
