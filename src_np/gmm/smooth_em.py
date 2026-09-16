"""Faithful NumPy port of the Smoothed-EM interactive laboratory.

The method is deliberately overcomplete: it estimates the number of active
components by simplex weight pruning rather than receiving the oracle number of
components.  It uses localized zeroth and vector first moments.  The two modes
are the Section 2.5 homogeneous and Section 3 inhomogeneous procedures from
the original interactive prototype.
"""

from __future__ import annotations

import numpy as np

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.iteration import IterationSnapshot, MixtureParameters


class SmoothEMGaussianMixtureModel:
    """Overcomplete smoothed moment/EM estimator for spherical GMMs.

    The reference procedure initializes one component at every design point:
    ``initial_components=None`` therefore gives ``K = J`` and ``m_k = x_k``.
    Supplying a value is available only as an explicit ablation override.
    """

    def __init__(
        self,
        initial_components=None,
        mode="inhomogeneous",
        target_neighbor_fraction=1.0 / 6.0,
        target_neighbor_count=None,
        n_design_points=400,
        base_variance=1.0,
        regularization=0.1,
        max_steps=None,
        internal_steps=5,
        n_directions=5,
        min_variance=0.05,
        max_variance=8.0,
        active_weight_tol=1e-8,
        objective_rtol=0.0,
        objective_atol=0.0,
        convergence_patience=2,
        weight_steps=80,
        data_batch_size=4096,
        design_batch_size=512,
        random_state=None,
        # Accepted only for compatibility with older UI declarations.  The
        # laboratory procedure prunes zero weights and never merges components.
        merge_threshold=None,
    ):
        self.initial_components = initial_components
        self.mode = str(mode)
        self.target_neighbor_fraction = float(target_neighbor_fraction)
        self.target_neighbor_count = target_neighbor_count
        self.n_design_points = int(n_design_points)
        self.base_variance = float(base_variance)
        self.regularization = float(regularization)
        self.max_steps = max_steps
        self.internal_steps = int(internal_steps)
        self.n_directions = int(n_directions)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)
        self.active_weight_tol = float(active_weight_tol)
        self.objective_rtol = float(objective_rtol)
        self.objective_atol = float(objective_atol)
        self.convergence_patience = int(convergence_patience)
        self.weight_steps = int(weight_steps)
        self.data_batch_size = int(data_batch_size)
        self.design_batch_size = int(design_batch_size)
        self.random_state = random_state
        self.merge_threshold = merge_threshold
        self._validate_configuration()

    def _validate_configuration(self):
        if self.initial_components is not None:
            self.initial_components = int(self.initial_components)
            if self.initial_components < 1:
                raise ValueError("initial_components must be positive or None")
        if self.mode not in ("homogeneous", "inhomogeneous"):
            raise ValueError("mode must be 'homogeneous' or 'inhomogeneous'")
        if (
            not np.isfinite(self.target_neighbor_fraction)
            or not 0 < self.target_neighbor_fraction < 1
        ):
            raise ValueError("target_neighbor_fraction must lie in (0, 1)")
        if self.target_neighbor_count is not None:
            self.target_neighbor_count = float(self.target_neighbor_count)
            if (
                not np.isfinite(self.target_neighbor_count)
                or self.target_neighbor_count <= 0
            ):
                raise ValueError("target_neighbor_count must be finite and positive")
        if self.max_steps is None:
            self.max_steps = 7 if self.mode == "inhomogeneous" else 3
        self.max_steps = int(self.max_steps)
        values = (
            self.regularization,
            self.base_variance,
            self.active_weight_tol,
            self.objective_rtol,
            self.objective_atol,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "base_variance, regularization, tolerances, and active_weight_tol must be finite and non-negative"
            )
        if self.base_variance <= 0:
            raise ValueError("base_variance must be positive")
        if self.n_design_points < 1 or self.max_steps < 0 or self.internal_steps < 1:
            raise ValueError("invalid design or iteration count")
        if (
            self.n_directions < 1
            or self.weight_steps < 1
            or self.data_batch_size < 1
            or self.design_batch_size < 1
        ):
            raise ValueError("batch sizes and iteration counts must be positive")
        if self.convergence_patience < 1:
            raise ValueError("convergence_patience must be at least one")
        if (
            not np.isfinite(self.min_variance)
            or not np.isfinite(self.max_variance)
            or not 0 < self.min_variance <= self.max_variance
        ):
            raise ValueError("variance bounds must satisfy 0 < min <= max")

    @staticmethod
    def _project_simplex(values):
        values = np.asarray(values, dtype=float).reshape(-1)
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
        minimum_distance = np.sum((points - seeds[0]) ** 2, axis=1)
        for _ in range(1, count):
            total = float(np.sum(minimum_distance))
            index = (
                int(rng.integers(points.shape[0]))
                if total <= np.finfo(float).tiny
                else int(rng.choice(points.shape[0], p=minimum_distance / total))
            )
            seeds.append(points[index].copy())
            minimum_distance = np.minimum(
                minimum_distance, np.sum((points - points[index]) ** 2, axis=1)
            )
        return np.asarray(seeds, dtype=float)

    def _initial_component_count(self, design):
        if self.initial_components is not None:
            return min(self.initial_components, design.shape[0])
        return design.shape[0]

    def _design(self, X, rng):
        indices = rng.choice(
            X.shape[0], size=min(self.n_design_points, X.shape[0]), replace=False
        )
        self.design_indices_ = indices.copy()
        return X[indices].copy()

    def _bandwidth(self, X, design):
        target = self.target_neighbor_count
        if target is None:
            target = max(
                1.0,
                min(float(X.shape[0] - 1), self.target_neighbor_fraction * X.shape[0]),
            )
        return float(
            select_s_values_by_average_kernel_count(
                X,
                1,
                target_neighbor_counts=[target],
                test_centers=design,
                max_bisection_steps=50,
            )[0]
        )

    def _moment_design(self, X, design, s):
        """Return the prototype's localized zeroth and vector first moments."""
        zeroth = np.zeros(design.shape[0], dtype=float)
        first = np.zeros_like(design, dtype=float)
        s2 = s * s
        for c0 in range(0, design.shape[0], self.design_batch_size):
            c1 = min(c0 + self.design_batch_size, design.shape[0])
            centers = design[c0:c1]
            c0_sum = np.zeros(c1 - c0, dtype=float)
            c1_sum = np.zeros_like(centers)
            for x0 in range(0, X.shape[0], self.data_batch_size):
                points = X[x0 : x0 + self.data_batch_size]
                difference = points[:, None, :] - centers[None, :, :]
                kernel = np.exp(-0.5 * np.sum(difference * difference, axis=2) / s2)
                c0_sum += np.sum(kernel, axis=0)
                c1_sum += np.sum(difference * kernel[:, :, None], axis=0)
            zeroth[c0:c1] = c0_sum / X.shape[0]
            first[c0:c1] = c1_sum / X.shape[0]
        return zeroth, first

    @staticmethod
    def _kernel_matrix(design, means, variances, s):
        difference = means[None, :, :] - design[:, None, :]
        return np.exp(
            -0.5
            * np.sum(difference * difference, axis=2)
            / (variances[None, :] + s * s)
        )

    def _solve_weights(self, matrix, target, weights, penalty):
        n_rows = matrix.shape[0]
        frobenius = float(np.sum(matrix * matrix))
        step = 0.75 / max(1e-8, 2.0 * frobenius / n_rows)
        candidate = weights.copy()
        for _ in range(self.weight_steps):
            residual = matrix @ candidate - target
            gradient = 2.0 * (matrix.T @ residual) / n_rows
            gradient -= penalty / n_rows * (np.log(candidate + 1e-12) + 1.0)
            candidate = self._project_simplex(candidate - step * gradient)
        return candidate

    def _prune(self, means, variances, weights):
        keep = np.flatnonzero(weights > self.active_weight_tol)
        if keep.size < min(2, weights.size):
            keep = np.argsort(-weights)[: min(2, weights.size)]
        weights = weights[keep]
        weights /= np.sum(weights)
        return means[keep], variances[keep], weights, keep

    @staticmethod
    def _solve_mean_system(A, target, design):
        K = A.shape[1]
        row_sum = np.sum(A, axis=1)
        rhs = A.T @ (target + row_sum[:, None] * design)
        ata = A.T @ A
        ridge = 2e-3 * (np.trace(ata) / max(K, 1) + 1.0)
        return np.linalg.solve(ata + ridge * np.eye(K), rhs)

    def _homogeneous_internal_step(self, design, C, Z, s, means, weights):
        variances = np.full(len(weights), self.base_variance, dtype=float)
        W = self._kernel_matrix(design, means, variances, s)
        nu = float(
            np.power(1.0 + self.base_variance / (s * s), design.shape[1] / 2.0)
        )
        weights = self._solve_weights(W, nu * C, weights, self.regularization)
        means, variances, weights, _ = self._prune(means, variances, weights)
        A = (
            self._kernel_matrix(design, means, variances, s)
            * weights[None, :]
            * (s * s / (self.base_variance + s * s))
        )
        means = self._solve_mean_system(A, nu * Z, design)
        return means, weights, nu

    def _inhomogeneous_internal_step(
        self, design, C, Z, s, means, variances, weights, effective_dimension
    ):
        W = self._kernel_matrix(design, means, variances, s)
        alpha = np.power(1.0 + variances / (s * s), -0.5 * effective_dimension)
        gamma = alpha / (1.0 + variances / (s * s))
        nu = np.power(
            1.0 + self.base_variance / (s * s), effective_dimension / 2.0
        )
        B = W * alpha[None, :]
        weights = self._solve_weights(B, C, weights, self.regularization / (nu * nu))
        means, variances, weights, keep = self._prune(means, variances, weights)
        A = (
            self._kernel_matrix(design, means, variances, s)
            * weights[None, :]
            * gamma[keep][None, :]
        )
        means = self._solve_mean_system(A, Z, design)
        return (
            means,
            variances,
            weights,
            float(nu),
            float(self.regularization / (nu * nu)),
        )

    def _adaptive_labeling(
        self, X, means, variances, weights, effective_dimension, directions
    ):
        variances = np.clip(variances, self.min_variance, self.max_variance)
        difference = X[:, None, :] - means[None, :, :]
        log_probability = np.log(np.maximum(weights, 1e-300))[None, :] - 0.5 * (
            effective_dimension * np.log(variances)[None, :]
            + np.sum(difference * difference, axis=2) / variances[None, :]
        )
        log_probability -= np.max(log_probability, axis=1, keepdims=True)
        responsibilities = np.exp(log_probability)
        responsibilities /= np.maximum(
            np.sum(responsibilities, axis=1, keepdims=True), 1e-300
        )
        nk = np.sum(responsibilities, axis=0)
        means = (responsibilities.T @ X) / np.maximum(nk[:, None], 1e-12)
        projected = np.einsum(
            "nkd,ld->nkl", X[:, None, :] - means[None, :, :], directions
        )
        variances = np.sum(
            responsibilities[:, :, None] * projected * projected, axis=(0, 2)
        ) / np.maximum(nk * directions.shape[0], 1e-12)
        return (
            means,
            np.clip(variances, self.min_variance, self.max_variance),
            nk / np.sum(nk),
        )

    def _effective_dimension(self, design, C, means, variances, weights, s):
        upper = max(20.0, float(np.ceil(1.5 * design.shape[1])))
        coarse_step = max(0.5, design.shape[1] / 40.0)
        log_factors = np.log1p(variances / (s * s))
        kernel = self._kernel_matrix(design, means, variances, s)

        def loss(candidate):
            fitted = kernel @ (weights * np.exp(-0.5 * candidate * log_factors))
            return float(np.sum((fitted - C) ** 2))

        coarse = np.arange(2.0, upper + 1e-12, coarse_step)
        best = float(coarse[np.argmin([loss(value) for value in coarse])])
        fine_step = coarse_step / 5.0
        fine = np.arange(
            max(2.0, best - coarse_step),
            min(upper, best + coarse_step) + 1e-12,
            fine_step,
        )
        return float(fine[np.argmin([loss(value) for value in fine])])

    @staticmethod
    def _negative_log_likelihood(X, means, variances, weights):
        d = X.shape[1]
        difference = X[:, None, :] - means[None, :, :]
        log_density = np.log(np.maximum(weights, 1e-300))[None, :] - 0.5 * (
            d * np.log(2.0 * np.pi * variances)[None, :]
            + np.sum(difference * difference, axis=2) / variances[None, :]
        )
        maximum = np.max(log_density, axis=1, keepdims=True)
        return float(
            -np.mean(
                maximum[:, 0] + np.log(np.sum(np.exp(log_density - maximum), axis=1))
            )
        )

    def _diagnostics(
        self,
        X,
        design,
        C,
        Z,
        s,
        means,
        variances,
        weights,
        effective_dimension,
        nu=1.0,
        homogeneous_moments=False,
    ):
        kernel = self._kernel_matrix(design, means, variances, s)
        if homogeneous_moments:
            # Equations (2.2)--(2.3): the reported residuals must retain the
            # normalizing factor nu used by the homogeneous updates.
            zero_residual = kernel @ weights - nu * C
            A = kernel * weights[None, :] * (
                s * s / (self.base_variance + s * s)
            )
            first_residual = A @ means - (
                nu * Z + np.sum(A, axis=1)[:, None] * design
            )
            zero_loss = float(np.mean(zero_residual * zero_residual))
            first_loss = float(np.mean(first_residual * first_residual))
            return (
                zero_loss,
                first_loss,
                self._negative_log_likelihood(X, means, variances, weights),
            )
        alpha = np.power(1.0 + variances / (s * s), -0.5 * effective_dimension)
        gamma = alpha / (1.0 + variances / (s * s))
        zero_residual = kernel @ (weights * alpha) - C
        A = kernel * weights[None, :] * gamma[None, :]
        first_residual = A @ means - (Z + np.sum(A, axis=1)[:, None] * design)
        zero_loss = float(np.mean(zero_residual * zero_residual))
        first_loss = float(np.mean(first_residual * first_residual))
        return (
            zero_loss,
            first_loss,
            self._negative_log_likelihood(X, means, variances, weights),
        )

    def fit(self, X, iteration_callback=None, **dataset):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
            raise ValueError(
                "X must be a finite two-dimensional array with at least two observations"
            )
        rng = np.random.default_rng(self.random_state)
        design = self._design(X, rng)
        self.s_ = self._bandwidth(X, design)
        self.design_points_ = design.copy()
        self.observed_zeroth_moments_, self.observed_moments_ = self._moment_design(
            X, design, self.s_
        )
        homogeneous_nu = float(
            np.power(
                1.0 + self.base_variance / (self.s_ * self.s_), X.shape[1] / 2.0
            )
        )
        initial_count = self._initial_component_count(design)
        means = self._farthest_seeds(design, initial_count, rng)
        variances = np.full(initial_count, self.base_variance, dtype=float)
        weights = np.full(initial_count, 1.0 / initial_count)
        self.history_ = []
        self.objective_history_ = []
        self.negative_log_likelihood_history_ = []

        def report(
            iteration,
            phase,
            *,
            effective_dimension,
            nu=None,
            penalty=None,
            external=0,
            internal=0,
            homogeneous_moments=False,
        ):
            zero_loss, first_loss, nll = self._diagnostics(
                X,
                design,
                self.observed_zeroth_moments_,
                self.observed_moments_,
                self.s_,
                means,
                variances,
                weights,
                effective_dimension,
                1.0 if nu is None else nu,
                homogeneous_moments,
            )
            loss = zero_loss + first_loss
            snapshot = IterationSnapshot(
                iteration=iteration,
                loss=loss,
                parameters=MixtureParameters.spherical(
                    means, weights, np.sqrt(variances)
                ),
                phase=phase,
                losses={
                    "zeroth_moment": zero_loss,
                    "first_moment": first_loss,
                    "negative_log_likelihood": nll,
                },
                metadata={
                    "bandwidth": self.s_,
                    "mode": self.mode,
                    "n_components": len(means),
                    "effective_dimension": effective_dimension,
                    "nu": nu,
                    "penalty": penalty,
                    "external_step": external,
                    "internal_step": internal,
                },
            )
            self.history_.append(snapshot)
            self.objective_history_.append(loss)
            self.negative_log_likelihood_history_.append(nll)
            if iteration_callback is not None:
                iteration_callback(snapshot)

        iteration = 0
        report(
            iteration,
            "initialization",
            effective_dimension=float(X.shape[1]),
            nu=homogeneous_nu,
            homogeneous_moments=True,
        )

        effective_dimension = float(X.shape[1])
        if self.mode == "homogeneous":
            for external in range(1, self.max_steps + 1):
                for internal in range(1, self.internal_steps + 1):
                    means, weights, nu = self._homogeneous_internal_step(
                        design,
                        self.observed_zeroth_moments_,
                        self.observed_moments_,
                        self.s_,
                        means,
                        weights,
                    )
                    variances = np.full(len(weights), self.base_variance)
                    iteration += 1
                    report(
                        iteration,
                        "optimization",
                        effective_dimension=float(X.shape[1]),
                        nu=nu,
                        penalty=self.regularization,
                        external=external,
                        internal=internal,
                        homogeneous_moments=True,
                    )
                means, variances, weights = self._adaptive_labeling(
                    X, means, variances, weights, X.shape[1], np.eye(X.shape[1])
                )
                variances = np.full(len(weights), self.base_variance)
                iteration += 1
                report(
                    iteration,
                    "labeling",
                    effective_dimension=float(X.shape[1]),
                    nu=nu,
                    external=external,
                    internal=self.internal_steps,
                    homogeneous_moments=True,
                )
        elif self.max_steps > 0:
            # The prototype intentionally starts Section 3 with one complete
            # Section 2.5 block before estimating heterogeneous variances.
            for internal in range(1, self.internal_steps + 1):
                means, weights, nu = self._homogeneous_internal_step(
                    design,
                    self.observed_zeroth_moments_,
                    self.observed_moments_,
                    self.s_,
                    means,
                    weights,
                )
                variances = np.full(len(weights), self.base_variance)
                iteration += 1
                report(
                    iteration,
                    "warmup",
                    effective_dimension=float(X.shape[1]),
                    nu=nu,
                    penalty=self.regularization,
                    external=1,
                    internal=internal,
                    homogeneous_moments=True,
                )
            directions = rng.normal(
                size=(min(self.n_directions, X.shape[1]), X.shape[1])
            )
            directions /= np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), np.finfo(float).tiny
            )
            means, variances, weights = self._adaptive_labeling(
                X, means, variances, weights, X.shape[1], directions
            )
            effective_dimension = self._effective_dimension(
                design,
                self.observed_zeroth_moments_,
                means,
                variances,
                weights,
                self.s_,
            )
            iteration += 1
            report(
                iteration,
                "labeling",
                effective_dimension=effective_dimension,
                nu=np.power(
                    1.0 + self.base_variance / (self.s_ * self.s_),
                    effective_dimension / 2.0,
                ),
                external=1,
                internal=self.internal_steps,
            )
            for external in range(2, self.max_steps + 1):
                for internal in range(1, self.internal_steps + 1):
                    means, variances, weights, nu, penalty = (
                        self._inhomogeneous_internal_step(
                            design,
                            self.observed_zeroth_moments_,
                            self.observed_moments_,
                            self.s_,
                            means,
                            variances,
                            weights,
                            effective_dimension,
                        )
                    )
                    iteration += 1
                    report(
                        iteration,
                        "optimization",
                        effective_dimension=effective_dimension,
                        nu=nu,
                        penalty=penalty,
                        external=external,
                        internal=internal,
                    )
                means, variances, weights = self._adaptive_labeling(
                    X, means, variances, weights, effective_dimension, directions
                )
                effective_dimension = self._effective_dimension(
                    design,
                    self.observed_zeroth_moments_,
                    means,
                    variances,
                    weights,
                    self.s_,
                )
                iteration += 1
                report(
                    iteration,
                    "labeling",
                    effective_dimension=effective_dimension,
                    nu=np.power(
                        1.0 + self.base_variance / (self.s_ * self.s_),
                        effective_dimension / 2.0,
                    ),
                    external=external,
                    internal=self.internal_steps,
                )

        self.means_ = np.asarray(means, dtype=float)
        self.sigmas_ = np.sqrt(np.asarray(variances, dtype=float))
        self.weights_ = np.asarray(weights, dtype=float)
        self.weights_ /= np.sum(self.weights_)
        self.effective_dimension_ = float(
            effective_dimension if self.mode == "inhomogeneous" else X.shape[1]
        )
        self.amplitudes_ = self.weights_ * np.exp(
            -0.5
            * self.effective_dimension_
            * np.log1p(variances / (self.s_ * self.s_))
        )
        self.objective_ = float(self.objective_history_[-1])
        self.negative_log_likelihood_ = float(self.negative_log_likelihood_history_[-1])
        self.n_iter_ = iteration
        self.converged_ = False
        return self

    def params_dict(self):
        if not hasattr(self, "means_"):
            raise RuntimeError("fit must be called before params_dict")
        return {
            "means": self.means_.copy(),
            "weights": self.weights_.copy(),
            "covariances": np.square(self.sigmas_)[:, None, None]
            * np.eye(self.means_.shape[1])[None, :, :],
        }


__all__ = ["SmoothEMGaussianMixtureModel"]
