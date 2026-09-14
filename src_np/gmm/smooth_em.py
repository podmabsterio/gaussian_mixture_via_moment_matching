"""Smoothed EM estimators ported from the ``smooth_em_v4.html`` prototype.

The implementation intentionally keeps the prototype's two research modes:
``homogeneous`` uses localized first moments and a profiled scale coefficient,
while ``inhomogeneous`` alternates the same sparse moment step with an
isotropic variance/weight update from randomized one-dimensional projections.
It is an overcomplete estimator: ``n_components`` is accepted for the common
repository API but is not used to seed the initial number of components.
"""

from __future__ import annotations

import numpy as np

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.iteration import IterationSnapshot, MixtureParameters


class SmoothEMGaussianMixtureModel:
    """Overcomplete smoothed-EM estimator for spherical Gaussian mixtures."""

    def __init__(
        self,
        n_components,
        mode="inhomogeneous",
        target_neighbor_fraction=1.0 / 6.0,
        target_neighbor_count=None,
        n_design_points=400,
        regularization=6.0,
        max_steps=10,
        internal_steps=5,
        n_directions=10,
        min_variance=0.0025,
        max_variance=8.0,
        merge_threshold=0.65,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.mode = str(mode)
        self.target_neighbor_fraction = float(target_neighbor_fraction)
        self.target_neighbor_count = target_neighbor_count
        self.n_design_points = int(n_design_points)
        self.regularization = float(regularization)
        self.max_steps = int(max_steps)
        self.internal_steps = int(internal_steps)
        self.n_directions = int(n_directions)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.merge_threshold = float(merge_threshold)
        self.random_state = random_state
        self._validate_configuration()

    def _validate_configuration(self):
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.mode not in ("homogeneous", "inhomogeneous"):
            raise ValueError("mode must be 'homogeneous' or 'inhomogeneous'")
        if not 0 < self.target_neighbor_fraction < 1:
            raise ValueError("target_neighbor_fraction must lie in (0, 1)")
        if self.target_neighbor_count is not None and self.target_neighbor_count <= 0:
            raise ValueError("target_neighbor_count must be positive")
        if self.n_design_points < 1 or self.regularization < 0:
            raise ValueError("invalid design or regularization setting")
        if self.max_steps < 0 or self.internal_steps < 1 or self.n_directions < 1:
            raise ValueError("iteration counts must be positive")
        if not 0 < self.min_variance <= self.max_variance:
            raise ValueError("variance bounds must satisfy 0 < min <= max")
        if self.merge_threshold < 0:
            raise ValueError("merge_threshold must be non-negative")

    @staticmethod
    def _project_simplex(values):
        values = np.asarray(values, dtype=float)
        order = np.sort(values)[::-1]
        cssv = np.cumsum(order)
        rho = np.nonzero(order - (cssv - 1) / (np.arange(values.size) + 1) > 0)[0]
        if rho.size == 0:
            return np.full_like(values, 1.0 / values.size)
        theta = (cssv[rho[-1]] - 1.0) / (rho[-1] + 1)
        return np.maximum(values - theta, 0.0)

    @staticmethod
    def _farthest_seeds(points, count, rng):
        count = min(int(count), points.shape[0])
        seeds = [points[rng.integers(points.shape[0])].copy()]
        min_dist = np.sum((points - seeds[0]) ** 2, axis=1)
        for _ in range(1, count):
            total = float(np.sum(min_dist))
            if total <= np.finfo(float).tiny:
                index = int(rng.integers(points.shape[0]))
            else:
                index = int(rng.choice(points.shape[0], p=min_dist / total))
            seeds.append(points[index].copy())
            min_dist = np.minimum(min_dist, np.sum((points - points[index]) ** 2, axis=1))
        return np.asarray(seeds, dtype=float)

    def _design(self, X, rng):
        J = min(self.n_design_points, X.shape[0])
        indices = rng.choice(X.shape[0], size=J, replace=False)
        design = X[indices].copy()
        self.design_indices_ = indices
        return design

    def _bandwidth(self, X, design):
        if self.target_neighbor_count is None:
            target = max(1.0, min(float(X.shape[0] - 1), self.target_neighbor_fraction * X.shape[0]))
        else:
            target = float(self.target_neighbor_count)
        return float(select_s_values_by_average_kernel_count(
            X, 1, target_neighbor_counts=[target], test_centers=design,
            max_bisection_steps=50,
        )[0])

    def _moment_design(self, X, design, s):
        # The HTML prototype uses the full vector response
        # ``mean((x - c) K_s(x-c))``.  This is not the scalar directional
        # response exposed by ``compute_Z_moment`` in the moment-GMM model.
        difference = X[:, None, :] - design[None, :, :]
        kernel = np.exp(-0.5 * np.sum(difference * difference, axis=2) / (s * s))
        response = np.mean(difference * kernel[:, :, None], axis=0)
        return response, None

    @staticmethod
    def _merge(means, variances, weights, threshold):
        order = np.argsort(-weights)
        used = np.zeros(len(means), dtype=bool)
        new_means, new_variances, new_weights = [], [], []
        for index in order:
            if used[index]:
                continue
            distances = np.sum((means - means[index]) ** 2, axis=1)
            group = np.flatnonzero((~used) & (distances <= threshold * threshold))
            used[group] = True
            mass = max(float(np.sum(weights[group])), np.finfo(float).tiny)
            center = np.sum(weights[group, None] * means[group], axis=0) / mass
            variance = np.sum(weights[group] * (variances[group] + np.sum((means[group] - center) ** 2, axis=1) / means.shape[1])) / mass
            new_means.append(center)
            new_variances.append(variance)
            new_weights.append(mass)
        new_weights = np.asarray(new_weights, dtype=float)
        new_weights /= max(float(np.sum(new_weights)), np.finfo(float).tiny)
        return np.asarray(new_means), np.asarray(new_variances), new_weights

    @staticmethod
    def _snapshot(iteration, means, variances, weights, loss, phase, metadata=None):
        return IterationSnapshot(
            iteration=iteration,
            loss=float(loss),
            parameters=MixtureParameters.spherical(means, weights, np.sqrt(variances)),
            phase=phase,
            metadata=metadata or {},
        )

    def _moment_step(self, X, design, Z, s, means, variances, weights, rng):
        K, d = means.shape
        J = design.shape[0]
        effective_variances = np.ones(K) if self.mode == "homogeneous" else variances
        total_variance = effective_variances[None, :] + s * s
        diff = design[:, None, :] - means[None, :, :]
        kernels = np.exp(-0.5 * np.sum(diff * diff, axis=2) / total_variance)
        A = kernels * weights[None, :]
        n_j = np.sum(A, axis=1)
        ata = A.T @ A
        rhs = np.zeros((K, 2 * d))
        rhs[:, :d] = A.T @ (n_j[:, None] * design)
        rhs[:, d:] = A.T @ Z
        ridge = 2e-3 * (np.trace(ata) / max(K, 1) + 1.0)
        solution = np.linalg.solve(ata + ridge * np.eye(K), rhs)
        mean_base, moment_base = solution[:, :d], solution[:, d:]
        residual_base = A @ moment_base - Z
        residual_mean = np.einsum("jk,kd->jd", A, mean_base) - n_j[:, None] * design
        numerator = float(np.sum(residual_mean * residual_base))
        denominator = float(np.sum(residual_base * residual_base))
        coefficient = np.clip(-numerator / denominator if denominator > 1e-14 else 1.0, 1e-4, 1e4)
        new_means = 0.7 * means + 0.3 * (mean_base + coefficient * moment_base)

        norm_d = float(np.sum(kernels * kernels * np.sum((new_means[None] - design[:, None]) ** 2, axis=2)))
        step = 0.55 / (2.0 * norm_d / max(J * d, 1) + 1e-6)
        penalty = max(self.regularization / max(J * d, 1), 0.0015 * self.regularization)
        new_weights = weights.copy()
        for _ in range(45):
            residual = np.einsum("jk,k,kd->jd", kernels, new_weights, new_means) - n_j[:, None] * design - coefficient * Z
            gradient = np.sum(2.0 * kernels[:, :, None] * (new_means[None] - design[:, None]) * residual[:, None, :], axis=(0, 2)) / max(J * d, 1)
            gradient -= penalty * (np.log(new_weights + 1e-10) + 1.0)
            new_weights = self._project_simplex(new_weights - step * gradient)

        threshold = max(1.0 / X.shape[0], 0.0025 * self.regularization)
        keep = np.flatnonzero(new_weights > threshold)
        if keep.size < 2 and K >= 2:
            keep = np.argsort(-new_weights)[:2]
        means, variances, weights = new_means[keep], variances[keep], new_weights[keep]
        weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
        means, variances, weights = self._merge(means, variances, weights, self.merge_threshold * np.sqrt(d))
        return means, variances, weights, coefficient

    def _adaptive_labeling(self, X, means, variances, directions):
        diff = X[:, None, :] - means[None, :, :]
        likelihood = np.exp(-0.5 * np.sum(diff * diff, axis=2) / np.maximum(self.min_variance, variances)[None, :])
        likelihood /= np.maximum(np.sum(likelihood, axis=1, keepdims=True), 1e-300)
        nk = np.sum(likelihood, axis=0)
        means = (likelihood.T @ X) / np.maximum(nk[:, None], 1e-12)
        centered = X[:, None, :] - means[None, :, :]
        projected = np.einsum("nkd,ld->nkl", centered, directions)
        variances = np.sum(likelihood[:, :, None] * projected * projected, axis=(0, 2)) / np.maximum(nk * directions.shape[0], 1e-12)
        variances = np.clip(variances, self.min_variance, self.max_variance)
        weights = nk / max(float(np.sum(nk)), np.finfo(float).tiny)
        return means, variances, weights

    def fit(self, X, iteration_callback=None, **dataset):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a non-empty finite two-dimensional array")
        rng = np.random.default_rng(self.random_state)
        design = self._design(X, rng)
        self.s_ = self._bandwidth(X, design)
        Z, directions = self._moment_design(X, design, self.s_)
        K0 = min(design.shape[0], max(8, min(24, 2 * X.shape[1] + 4)))
        means = self._farthest_seeds(design, K0, rng)
        variances = np.ones(K0, dtype=float)
        weights = np.full(K0, 1.0 / K0)
        self.history_ = []

        def report(step, phase, loss, metadata=None):
            snapshot = self._snapshot(step, means, variances, weights, loss, phase, metadata)
            self.history_.append(snapshot)
            if iteration_callback is not None:
                iteration_callback(snapshot)

        report(0, "initialization", 0.0, {"bandwidth": self.s_, "mode": self.mode, "n_components": len(means)})
        for outer in range(self.max_steps):
            if self.mode == "inhomogeneous" and outer > 0:
                weights *= ((1.0 + self.s_ * self.s_) / (variances + self.s_ * self.s_)) ** (X.shape[1] / 2 + 1)
                weights /= max(float(np.sum(weights)), np.finfo(float).tiny)
            for _ in range(self.internal_steps if self.mode == "inhomogeneous" else 1):
                means, variances, weights, coefficient = self._moment_step(X, design, Z, self.s_, means, variances, weights, rng)
            if self.mode == "inhomogeneous":
                directions = rng.normal(size=(self.n_directions, X.shape[1]))
                directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), np.finfo(float).tiny)
                means, variances, weights = self._adaptive_labeling(X, means, variances, directions)
            report(outer + 1, "optimization", 0.0, {"bandwidth": self.s_, "coefficient": float(coefficient), "mode": self.mode, "n_components": len(means)})

        if self.mode == "homogeneous":
            diff = X[:, None, :] - means[None, :, :]
            likelihood = np.exp(-0.5 * np.sum(diff * diff, axis=2))
            likelihood /= np.maximum(np.sum(likelihood, axis=1, keepdims=True), 1e-300)
            nk = np.sum(likelihood, axis=0)
            means = (likelihood.T @ X) / np.maximum(nk[:, None], 1e-12)
            weights = nk / max(float(np.sum(nk)), np.finfo(float).tiny)
            variances = np.ones(len(means))
            report(self.max_steps + 1, "polish", 0.0, {"bandwidth": self.s_, "mode": self.mode})

        self.means_ = np.asarray(means, dtype=float)
        self.sigmas_ = np.sqrt(np.asarray(variances, dtype=float))
        self.weights_ = np.asarray(weights, dtype=float)
        self.weights_ /= np.sum(self.weights_)
        self.amplitudes_ = self.weights_ * np.exp(-0.5 * X.shape[1] * np.log1p(variances / (self.s_ * self.s_)))
        self.objective_ = 0.0
        return self

    def params_dict(self):
        if not hasattr(self, "means_"):
            raise RuntimeError("fit must be called before params_dict")
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": np.square(self.sigmas_)[:, None, None]
            * np.eye(self.means_.shape[1], dtype=float)[None, :, :],
        }


__all__ = ["SmoothEMGaussianMixtureModel"]
