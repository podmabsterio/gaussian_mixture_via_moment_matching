from dataclasses import dataclass
from typing import Optional
from src.data_generation.dataclass import MixtureSpec, CenterResult

import numpy as np


Array = np.ndarray


class CovarianceSamplerBase:
    """Base class for covariance samplers."""

    def sample(
        self,
        spec: MixtureSpec,
        centers: CenterResult,
        rng: np.random.Generator,
    ) -> Array:
        """Sample covariance matrices."""
        raise NotImplementedError


@dataclass(slots=True)
class IsotropicCovarianceSampler(CovarianceSamplerBase):
    """Samples spherical covariance matrices."""

    base_scale: Optional[float] = None
    variance_spread: Optional[float] = None

    def sample(
        self,
        spec: MixtureSpec,
        centers: CenterResult,
        rng: np.random.Generator,
    ) -> Array:
        """Sample covariances of the form sigma_k^2 I."""
        base_scale = spec.base_scale if self.base_scale is None else self.base_scale
        spread = (
            spec.variance_spread
            if self.variance_spread is None
            else self.variance_spread
        )

        scales = self._sample_component_scales(
            n_components=spec.n_components,
            base_scale=base_scale,
            spread=spread,
            rng=rng,
        )

        identity = np.eye(spec.n_features)
        return scales[:, None, None] ** 2 * identity[None, :, :]

    def _sample_component_scales(
        self,
        n_components: int,
        base_scale: float,
        spread: float,
        rng: np.random.Generator,
    ) -> Array:
        """Sample component-wise standard deviations."""
        log_factors = rng.uniform(-spread, spread, size=n_components)
        return base_scale * np.exp(log_factors)


@dataclass(slots=True)
class AnisotropicCovarianceSampler(CovarianceSamplerBase):
    """Samples full anisotropic covariance matrices."""

    base_scale: Optional[float] = None
    variance_spread: Optional[float] = None
    anisotropy: Optional[float] = None

    def sample(
        self,
        spec: MixtureSpec,
        centers: CenterResult,
        rng: np.random.Generator,
    ) -> Array:
        """Sample covariances with random eigenvectors and eigenvalues."""
        base_scale = spec.base_scale if self.base_scale is None else self.base_scale
        spread = (
            spec.variance_spread
            if self.variance_spread is None
            else self.variance_spread
        )
        anisotropy = spec.anisotropy if self.anisotropy is None else self.anisotropy

        scales = self._sample_component_scales(
            n_components=spec.n_components,
            base_scale=base_scale,
            spread=spread,
            rng=rng,
        )

        covariances = np.empty(
            (spec.n_components, spec.n_features, spec.n_features),
            dtype=float,
        )

        for component_id, scale in enumerate(scales):
            eigenvalues = self._sample_eigenvalues(
                n_features=spec.n_features,
                scale=scale,
                anisotropy=anisotropy,
                rng=rng,
            )
            rotation = self._sample_orthogonal_matrix(spec.n_features, rng)
            covariances[component_id] = rotation @ np.diag(eigenvalues) @ rotation.T

        return covariances

    def _sample_component_scales(
        self,
        n_components: int,
        base_scale: float,
        spread: float,
        rng: np.random.Generator,
    ) -> Array:
        """Sample component-wise global scales."""
        log_factors = rng.uniform(-spread, spread, size=n_components)
        return base_scale * np.exp(log_factors)

    def _sample_eigenvalues(
        self,
        n_features: int,
        scale: float,
        anisotropy: float,
        rng: np.random.Generator,
    ) -> Array:
        """Sample eigenvalues with controlled condition number."""
        log_range = np.log(anisotropy)
        log_eigenvalues = rng.uniform(
            low=-0.5 * log_range,
            high=0.5 * log_range,
            size=n_features,
        )
        return scale**2 * np.exp(log_eigenvalues)

    def _sample_orthogonal_matrix(
        self,
        n_features: int,
        rng: np.random.Generator,
    ) -> Array:
        """Sample a random orthogonal matrix."""
        raw_matrix = rng.normal(size=(n_features, n_features))
        q, r = np.linalg.qr(raw_matrix)

        # Фиксируем знаки, чтобы QR не вносил лишнюю систематичность.
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0

        return q * signs


@dataclass(slots=True)
class TwoScaleSubspaceCovarianceSampler(CovarianceSamplerBase):
    """Samples covariances with two scales: inside and outside a subspace."""

    parallel_scale: Optional[float] = None
    orthogonal_scale: Optional[float] = None
    variance_spread: Optional[float] = None
    shared_across_components: bool = True

    def sample(
        self,
        spec: MixtureSpec,
        centers: CenterResult,
        rng: np.random.Generator,
    ) -> Array:
        """Sample two-scale subspace covariance matrices."""
        if centers.subspace_basis is None:
            raise ValueError(
                "TwoScaleSubspaceCovarianceSampler requires subspace_basis."
            )

        basis = centers.subspace_basis
        base_parallel = (
            spec.base_scale if self.parallel_scale is None else self.parallel_scale
        )
        base_orthogonal = self._get_orthogonal_scale(spec, base_parallel)
        spread = (
            spec.variance_spread
            if self.variance_spread is None
            else self.variance_spread
        )

        parallel_scales, orthogonal_scales = self._sample_scales(
            spec=spec,
            base_parallel=base_parallel,
            base_orthogonal=base_orthogonal,
            spread=spread,
            rng=rng,
        )

        projection = basis @ basis.T
        orthogonal_projection = np.eye(spec.n_features) - projection

        covariances = np.empty(
            (spec.n_components, spec.n_features, spec.n_features),
            dtype=float,
        )

        for component_id in range(spec.n_components):
            covariances[component_id] = (
                parallel_scales[component_id] ** 2 * projection
                + orthogonal_scales[component_id] ** 2 * orthogonal_projection
            )

        return covariances

    def _get_orthogonal_scale(
        self,
        spec: MixtureSpec,
        base_parallel: float,
    ) -> float:
        """Return the default orthogonal scale."""
        if self.orthogonal_scale is not None:
            return self.orthogonal_scale

        return base_parallel / np.sqrt(spec.anisotropy)

    def _sample_scales(
        self,
        spec: MixtureSpec,
        base_parallel: float,
        base_orthogonal: float,
        spread: float,
        rng: np.random.Generator,
    ) -> tuple[Array, Array]:
        """Sample parallel and orthogonal scales."""
        if self.shared_across_components:
            parallel_scales = np.full(spec.n_components, base_parallel)
            orthogonal_scales = np.full(spec.n_components, base_orthogonal)
            return parallel_scales, orthogonal_scales

        log_factors = rng.uniform(-spread, spread, size=spec.n_components)
        factors = np.exp(log_factors)

        return base_parallel * factors, base_orthogonal * factors
