from dataclasses import dataclass
from typing import Optional
from src.data_generation.dataclass import MixtureSpec, CenterResult

import numpy as np


Array = np.ndarray


class CenterSamplerBase:
    """Base class for component center samplers."""

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> CenterResult:
        """Sample component centers."""
        raise NotImplementedError


@dataclass(slots=True)
class FullDimensionalCenterSampler(CenterSamplerBase):
    """Samples centers in the full ambient space."""

    center_scale: Optional[float] = None

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> CenterResult:
        """Sample centers in R^d."""
        scale = spec.base_scale if self.center_scale is None else self.center_scale

        means = rng.normal(
            loc=0.0,
            scale=scale,
            size=(spec.n_components, spec.n_features),
        )
        means = means - means.mean(axis=0, keepdims=True)

        return CenterResult(
            means=means,
            metadata={
                "center_sampler": self.__class__.__name__,
                "center_scale": scale,
            },
        )


@dataclass(slots=True)
class FixedDistanceCenterSampler(CenterSamplerBase):
    """Samples centers with fixed pairwise distance."""

    center_scale: Optional[float] = None

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> CenterResult:
        """Sample centers in R^d."""
        scale = spec.base_scale if self.center_scale is None else self.center_scale

        n_centers = spec.n_components
        if n_centers > spec.n_features + 1:
            raise ValueError("Cannot sample more than d+1 centers with fixed distance.")

        if n_centers == 1:
            means = np.zeros((1, spec.n_features))
        else:
            centered_identity = np.eye(n_centers) - (1.0 / n_centers)
            basis, _ = np.linalg.qr(centered_identity[:, :-1])
            means = np.zeros((n_centers, spec.n_features))
            means[:, : n_centers - 1] = basis * scale

        means = means - means.mean(axis=0, keepdims=True)

        return CenterResult(
            means=means,
            metadata={
                "center_sampler": self.__class__.__name__,
                "center_scale": scale,
            },
        )


@dataclass(slots=True)
class NearAffineSubspaceCenterSampler(CenterSamplerBase):
    """Samples centers near a low-dimensional affine subspace."""

    subspace_dim: Optional[int] = None
    center_scale: Optional[float] = None
    origin_scale: float = 0.0

    def sample(self, spec: MixtureSpec, rng: np.random.Generator) -> CenterResult:
        """Sample centers near an affine subspace."""
        r = spec.subspace_dim if self.subspace_dim is None else self.subspace_dim
        scale = spec.base_scale if self.center_scale is None else self.center_scale

        basis = self._sample_subspace_basis(spec.n_features, r, rng)
        origin = rng.normal(
            loc=0.0,
            scale=self.origin_scale,
            size=spec.n_features,
        )

        latent_means = rng.normal(
            loc=0.0,
            scale=scale,
            size=(spec.n_components, r),
        )
        latent_means = latent_means - latent_means.mean(axis=0, keepdims=True)

        means = origin + latent_means @ basis.T

        if spec.subspace_noise > 0.0:
            noise = self._sample_orthogonal_noise(
                n_components=spec.n_components,
                n_features=spec.n_features,
                basis=basis,
                noise_scale=spec.subspace_noise * scale,
                rng=rng,
            )
            means = means + noise

        return CenterResult(
            means=means,
            subspace_basis=basis,
            subspace_origin=origin,
            metadata={
                "center_sampler": self.__class__.__name__,
                "subspace_dim": r,
                "center_scale": scale,
                "origin_scale": self.origin_scale,
                "subspace_noise": spec.subspace_noise,
            },
        )

    def _sample_subspace_basis(
        self,
        n_features: int,
        subspace_dim: int,
        rng: np.random.Generator,
    ) -> Array:
        """Sample an orthonormal basis of a random subspace."""
        raw_basis = rng.normal(size=(n_features, subspace_dim))
        basis, _ = np.linalg.qr(raw_basis)
        return basis[:, :subspace_dim]

    def _sample_orthogonal_noise(
        self,
        n_components: int,
        n_features: int,
        basis: Array,
        noise_scale: float,
        rng: np.random.Generator,
    ) -> Array:
        """Sample noise in the orthogonal complement."""
        noise = rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=(n_components, n_features),
        )

        # Проекция убирает компоненту шума внутри выбранного подпространства.
        projection = noise @ basis @ basis.T
        return noise - projection
