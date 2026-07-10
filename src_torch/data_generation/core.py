from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from src.data_generation.component_generator import (
    ComponentSamplerBase,
    GaussianComponentSampler,
)
from src.data_generation.covarience_generator import CovarianceSamplerBase
from src.data_generation.means_generator import CenterSamplerBase
from src.data_generation.separation_calibrator import SeparationCalibrator
from src.data_generation.weight_generator import WeightSamplerBase
from src.data_generation.dataclass import MixtureSpec, CenterResult, GeneratedDataset


Array = np.ndarray


class MixtureDatasetGenerator:
    """Main pipeline for synthetic mixture generation."""

    def __init__(
        self,
        weight_sampler: WeightSamplerBase,
        center_sampler: CenterSamplerBase,
        covariance_sampler: CovarianceSamplerBase,
        component_sampler: ComponentSamplerBase = None,
    ):
        self.weight_sampler = weight_sampler
        self.center_sampler = center_sampler
        self.covariance_sampler = covariance_sampler
        if component_sampler is None:
            self.component_sampler = GaussianComponentSampler()
        else:
            self.component_sampler = component_sampler

    def generate(self, spec: MixtureSpec) -> GeneratedDataset:
        """Generate a synthetic mixture dataset."""
        rng = np.random.default_rng(spec.random_seed)
        self.separation_calibrator = SeparationCalibrator()
        weights = self.weight_sampler.sample(spec, rng)
        centers = self.center_sampler.sample(spec, rng)
        covariances = self.covariance_sampler.sample(spec, centers, rng)

        means = centers.means
        if self.separation_calibrator is not None:
            means = self.separation_calibrator.calibrate(
                means=means,
                covariances=covariances,
                target_separation=spec.separation,
            )

        counts = self._sample_component_counts(spec.n_samples, weights, rng)
        X, y = self._sample_observations(means, covariances, counts, rng)
        X, y = self._shuffle(X, y, rng)

        metadata = dict(centers.metadata)
        metadata["component_counts"] = counts

        return GeneratedDataset(
            X=X,
            y=y,
            means=means,
            covariances=covariances,
            weights=weights,
            subspace_basis=centers.subspace_basis,
            subspace_origin=centers.subspace_origin,
            spec=spec,
            metadata=metadata,
        )

    def _sample_component_counts(
        self,
        n_samples: int,
        weights: Array,
        rng: np.random.Generator,
    ) -> Array:
        """Sample how many observations come from each component."""
        return rng.multinomial(n_samples, weights)

    def _sample_observations(
        self,
        means: Array,
        covariances: Array,
        counts: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, Array]:
        """Sample observations and labels component by component."""
        X_parts: list[Array] = []
        y_parts: list[Array] = []

        for component_id, count in enumerate(counts):
            if count == 0:
                continue

            X_k = self.component_sampler.sample(
                mean=means[component_id],
                covariance=covariances[component_id],
                size=int(count),
                rng=rng,
            )
            y_k = np.full(int(count), component_id, dtype=int)

            X_parts.append(X_k)
            y_parts.append(y_k)

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _shuffle(
        self,
        X: Array,
        y: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, Array]:
        """Shuffle observations and labels with the same permutation."""
        permutation = rng.permutation(len(y))
        return X[permutation], y[permutation]
