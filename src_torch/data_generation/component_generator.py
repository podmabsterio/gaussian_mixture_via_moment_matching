import numpy as np
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

Array = np.ndarray


class ComponentSamplerBase:
    """Base class for component distribution samplers."""

    def sample(
        self,
        mean: Array,
        covariance: Array,
        size: int,
        rng: np.random.Generator,
    ) -> Array:
        """Sample observations from one component."""
        raise NotImplementedError


@dataclass(slots=True)
class GaussianComponentSampler(ComponentSamplerBase):
    """Samples observations from a Gaussian component."""

    def sample(
        self,
        mean: Array,
        covariance: Array,
        size: int,
        rng: np.random.Generator,
    ) -> Array:
        """Sample from N(mean, covariance)."""
        return rng.multivariate_normal(
            mean=mean,
            cov=covariance,
            size=size,
        )
