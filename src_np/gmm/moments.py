import inspect
from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.gmm.experimental_oracle_centers import (
    CENTER_ROLE_NEAR_MEAN,
    build_experimental_center_plan,
    label_observed_center_roles,
)
from src_np.kmeans_initialization import initialize_dimension_free_one_s_gmm
from src_np.optimization import (
    fit_dimension_free_moment_gmm,
    fit_formula_variance_moment_gmm,
    fit_joint_variance_moment_gmm,
)
from src_np.optimization.utils import RADIAL_SECOND_MOMENT
from src_np.test_functions_response import (
    compute_kernel_counts,
    compute_Z_moment,
    compute_Z_radial_second_moment,
)


class MomentGaussianMixtureModel:
    """Configurable dimension-free isotropic GMM moment estimator.

    ``n_*_moments`` is the number of probes of that order per base test center.
    Setting a count to zero disables the corresponding order.  The old NumPy
    estimators are special cases:

    * ``OneSGaussianMixtureModel``: only first-order moments,
      ``s_optimization='sequential'``, and
      ``normalize_moment_losses=False``;
    * ``MultiSGaussianMixtureModel``: the same moment configuration with
      ``s_optimization='joint'`` and ``geometry_optimization='joint'``.

    By default, all three directional moment orders are enabled, every order-r
    response is divided by ``s**r``, and each family is averaged by its own
    number of test functions before the weighted losses are added.  Thus the
    configured moment-family weights act on the dimensionless test functions

    ``K_s(x-c)``, ``<x-c,u> / s * K_s(x-c)``, and
    ``<x-c,u><x-c,v> / s**2 * K_s(x-c)``.

    Set ``n_radial_second_moments=1`` to add the direction-free test function
    ``||x-c||**2 / (d * s**2) * K_s(x-c)`` at every base center.  There is only
    one such probe per center, so the supported count is zero or one.  With
    ``compensated_radial_second_moments=True``, ``K_s(x-c)`` is subtracted;
    this makes the radial test function integrate to zero on :math:`R^d`.
    Its ``d * s**2`` normalization is always applied, independently of
    ``scale_normalize_moments``.

    ``optimizer_mode='legacy'`` keeps the original optimizer unchanged.  The
    experimental ``'formula_variance'`` mode performs a component mean-only
    step and then updates that component variance using equation (26) from the
    problem statement.  It evaluates the required empirical zeroth and radial
    moments at the current mean using the smallest fitted bandwidth.  The
    ``'joint_variance'`` mode completes a full component-wise mean sweep with
    every variance fixed and then jointly optimizes all log variances.  These
    separated modes intentionally support only the recent experimental setup:
    joint bandwidth fitting, simplex weights, component mean steps, no mean
    penalty, and no joint polish.

    Set ``scale_normalize_moments=False`` to reproduce the original unscaled
    moment objective.  In particular, exact compatibility with the old NumPy
    estimators additionally requires this flag to be disabled.

    ``test_center_fraction`` can robustify data-derived test centers.  Values
    below one retain ``ceil(fraction * n_samples)`` observations with the
    largest Gaussian-kernel soft neighbor counts, evaluated at the smallest
    fitted bandwidth.  ``n_test_centers`` is then applied to this retained
    candidate pool.  Additional centers sampled near component means are not
    filtered.

    With ``leave_one_out=True``, each data-derived test center excludes its
    source observation both from empirical moment responses and from automatic
    bandwidth selection.  Centers without a unique source observation are left
    unchanged.

    ``target_neighbor_fractions`` is the sample-size-adaptive counterpart of
    ``target_neighbor_counts``.  Each fraction is multiplied by the maximum
    available soft-neighbor count (``n`` or ``n - 1`` under data-centered
    leave-one-out) before bandwidth selection.  The two target specifications
    are mutually exclusive.
    """

    def __init__(
        self,
        n_components,
        s_values=None,
        target_neighbor_counts=None,
        target_neighbor_fractions=None,
        init="kmeans",
        n_init=1,
        n_zero_moments=1,
        n_first_moments=1,
        n_second_moments=1,
        second_moment_mode="same_direction",
        zero_moment_weight=1.0,
        first_moment_weight=1.0,
        second_moment_weight=1.0,
        normalize_moment_losses=True,
        scale_normalize_moments=True,
        amplitude_optimization="non_negative",
        s_optimization="sequential",
        geometry_optimization="component",
        optimizer_mode="legacy",
        test_center_mode="data",
        n_test_centers=None,
        test_center_fraction=1.0,
        test_center_noise_std="auto",
        r_tests_near_means=0.0,
        alpha_std_tests=0.1,
        mean_penalty_weight=0.0,
        objective_rtol=1e-6,
        objective_atol=0.0,
        convergence_patience=2,
        random_state=None,
        max_steps=50,
        geom_sweeps=1,
        ls_max_nfev=30,
        active_tol=1e-8,
        eta_step_bound=0.75,
        m_step_bound=None,
        joint_polish_sweeps=0,
        joint_polish_max_nfev=None,
        variance_formula_q_bounds=(1e-4, 1.0 - 1e-4),
        # Compatibility aliases for configuring the old model modes.
        include_base_kernel=None,
        num_directions=None,
        joint_optimization=None,
        leave_one_out=False,
        experimental_oracle_test_centers=False,
        experimental_uniform_center_fraction=0.05,
        n_radial_second_moments=0,
        radial_second_moment_weight=1.0,
        compensated_radial_second_moments=False,
    ):
        self.k = int(n_components)
        self.s_values = s_values
        self.target_neighbor_counts = target_neighbor_counts
        self.target_neighbor_fractions = (
            None
            if target_neighbor_fractions is None
            else tuple(
                np.asarray(target_neighbor_fractions, dtype=float).reshape(-1)
            )
        )
        self.init = init
        self.n_init = int(n_init)

        if include_base_kernel is not None:
            if n_zero_moments != 1 and int(n_zero_moments) != int(
                bool(include_base_kernel)
            ):
                raise ValueError(
                    "n_zero_moments and include_base_kernel specify different modes"
                )
            n_zero_moments = int(bool(include_base_kernel))
        if num_directions is not None:
            if n_first_moments != 1 and int(n_first_moments) != int(num_directions):
                raise ValueError(
                    "n_first_moments and num_directions specify different counts"
                )
            n_first_moments = int(num_directions)
        if joint_optimization is not None:
            alias_geometry = "joint" if joint_optimization else "component"
            if (
                geometry_optimization != "component"
                and geometry_optimization != alias_geometry
            ):
                raise ValueError(
                    "geometry_optimization and joint_optimization disagree"
                )
            geometry_optimization = alias_geometry

        self.n_zero_moments = int(n_zero_moments)
        self.n_first_moments = int(n_first_moments)
        self.n_second_moments = int(n_second_moments)
        self.n_radial_second_moments = int(n_radial_second_moments)
        self.compensated_radial_second_moments = bool(compensated_radial_second_moments)
        self.second_moment_mode = self._canonical_second_moment_mode(second_moment_mode)
        self.moment_weights = {
            0: float(zero_moment_weight),
            1: float(first_moment_weight),
            2: float(second_moment_weight),
            RADIAL_SECOND_MOMENT: float(radial_second_moment_weight),
        }
        self.normalize_moment_losses = bool(normalize_moment_losses)
        self.scale_normalize_moments = bool(scale_normalize_moments)
        self.amplitude_optimization = amplitude_optimization
        self.s_optimization = self._canonical_s_optimization(s_optimization)
        self.geometry_optimization = geometry_optimization
        self.optimizer_mode = self._canonical_optimizer_mode(optimizer_mode)
        self.test_center_mode = test_center_mode
        self.n_test_centers = None if n_test_centers is None else int(n_test_centers)
        self.test_center_fraction = float(test_center_fraction)
        self.test_center_noise_std = test_center_noise_std
        self.leave_one_out = bool(leave_one_out)
        # TODO(research): Temporary opt-in switch for the oracle negative-space
        # probe experiment.  Keep ordinary model behavior independent of dataset
        # contamination metadata and remove this hook after the mechanism test.
        self.experimental_oracle_test_centers = bool(experimental_oracle_test_centers)
        self.experimental_uniform_center_fraction = float(
            experimental_uniform_center_fraction
        )
        self.r_tests_near_means = float(r_tests_near_means)
        self.alpha_std_tests = float(alpha_std_tests)
        self.mean_penalty_weight = float(mean_penalty_weight)
        self.objective_rtol = float(objective_rtol)
        self.objective_atol = float(objective_atol)
        self.convergence_patience = int(convergence_patience)
        self.random_state = random_state
        self.max_steps = int(max_steps)
        self.geom_sweeps = int(geom_sweeps)
        self.ls_max_nfev = int(ls_max_nfev)
        self.active_tol = float(active_tol)
        self.eta_step_bound = eta_step_bound
        self.m_step_bound = m_step_bound
        self.joint_polish_sweeps = int(joint_polish_sweeps)
        self.joint_polish_max_nfev = (
            None if joint_polish_max_nfev is None else int(joint_polish_max_nfev)
        )
        self.variance_formula_q_bounds = tuple(
            np.asarray(variance_formula_q_bounds, dtype=float).reshape(-1)
        )

        self.means_ = None
        self.sigmas_ = None
        self.amplitudes_ = None
        self.weights_ = None
        self._initial_weights = None
        self._validate_configuration()
        self._set_explicit_initialization(init)

    @staticmethod
    def _canonical_second_moment_mode(mode):
        aliases = {
            "same": "same_direction",
            "same_direction": "same_direction",
            "squared": "same_direction",
            "isotropic": "same_direction",
            "independent": "two_directions",
            "two_directions": "two_directions",
            "cross": "two_directions",
            "anisotropic": "two_directions",
        }
        try:
            return aliases[mode]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "second_moment_mode must be 'same_direction' or 'two_directions'"
            ) from exc

    @staticmethod
    def _canonical_s_optimization(mode):
        aliases = {
            "sequential": "sequential",
            "one_s": "sequential",
            "joint": "joint",
            "all": "joint",
            "multi_s": "joint",
        }
        try:
            return aliases[mode]
        except (KeyError, TypeError) as exc:
            raise ValueError("s_optimization must be 'sequential' or 'joint'") from exc

    @staticmethod
    def _canonical_optimizer_mode(mode):
        aliases = {
            "legacy": "legacy",
            "joint_geometry": "legacy",
            "formula_variance": "formula_variance",
            "profiled_variance": "formula_variance",
            "joint_variance": "joint_variance",
            "alternating_joint_variance": "joint_variance",
        }
        try:
            return aliases[mode]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "optimizer_mode must be 'legacy', 'formula_variance', or "
                "'joint_variance'"
            ) from exc

    def _validate_configuration(self):
        if self.k < 1:
            raise ValueError("n_components must be positive")
        if self.n_init < 1:
            raise ValueError("n_init must be positive")
        counts = {
            0: self.n_zero_moments,
            1: self.n_first_moments,
            2: self.n_second_moments,
            RADIAL_SECOND_MOMENT: self.n_radial_second_moments,
        }
        if self.n_radial_second_moments not in (0, 1):
            raise ValueError("n_radial_second_moments must be zero or one")
        if any(count < 0 for count in counts.values()) or sum(counts.values()) == 0:
            raise ValueError("moment counts must be non-negative and not all zero")
        if any(
            not np.isfinite(weight) or weight < 0
            for weight in self.moment_weights.values()
        ):
            raise ValueError("moment weights must be finite and non-negative")
        if not any(
            count > 0 and self.moment_weights[family] > 0
            for family, count in counts.items()
        ):
            raise ValueError("at least one enabled moment needs a positive weight")
        if self.geometry_optimization not in ("component", "joint"):
            raise ValueError("geometry_optimization must be 'component' or 'joint'")
        if self.amplitude_optimization not in ("non_negative", "simplex"):
            raise ValueError(
                "amplitude_optimization must be 'non_negative' or 'simplex'"
            )
        if self.test_center_mode not in ("data", "noisy_data"):
            raise ValueError("test_center_mode must be 'data' or 'noisy_data'")
        if self.n_test_centers is not None and self.n_test_centers < 1:
            raise ValueError("n_test_centers must be positive or None")
        if self.target_neighbor_fractions is not None:
            fractions = np.asarray(self.target_neighbor_fractions, dtype=float)
            if (
                fractions.size == 0
                or not np.all(np.isfinite(fractions))
                or np.any(fractions <= 0)
                or np.any(fractions >= 1)
            ):
                raise ValueError(
                    "target_neighbor_fractions must contain values in (0, 1)"
                )
            if self.target_neighbor_counts is not None:
                raise ValueError(
                    "target_neighbor_counts and target_neighbor_fractions are "
                    "mutually exclusive"
                )
        if (
            not np.isfinite(self.test_center_fraction)
            or not 0 < self.test_center_fraction <= 1
        ):
            raise ValueError("test_center_fraction must lie in (0, 1]")
        if self.r_tests_near_means < 0 or self.alpha_std_tests < 0:
            raise ValueError("test-center sampling scales must be non-negative")
        if self.mean_penalty_weight < 0:
            raise ValueError("mean_penalty_weight must be non-negative")
        if self.objective_rtol < 0 or self.objective_atol < 0:
            raise ValueError("objective tolerances must be non-negative")
        if self.convergence_patience < 1:
            raise ValueError("convergence_patience must be at least one")
        if self.max_steps < 0 or self.geom_sweeps < 1 or self.ls_max_nfev < 1:
            raise ValueError("optimization iteration counts are invalid")
        if self.active_tol < 0:
            raise ValueError("active_tol must be non-negative")
        if self.eta_step_bound is not None and self.eta_step_bound <= 0:
            raise ValueError("eta_step_bound must be positive or None")
        if self.m_step_bound is not None and self.m_step_bound <= 0:
            raise ValueError("m_step_bound must be positive or None")
        if self.joint_polish_sweeps < 0:
            raise ValueError("joint_polish_sweeps must be non-negative")
        if self.joint_polish_max_nfev is not None and self.joint_polish_max_nfev < 1:
            raise ValueError("joint_polish_max_nfev must be positive or None")
        if self.optimizer_mode != "legacy":
            if self.s_optimization != "joint":
                raise ValueError(
                    "separated-variance optimizers require s_optimization='joint'"
                )
            if self.amplitude_optimization != "simplex":
                raise ValueError(
                    "separated-variance optimizers require "
                    "amplitude_optimization='simplex'"
                )
            if self.geometry_optimization != "component":
                raise ValueError(
                    "separated-variance optimizers require "
                    "geometry_optimization='component'"
                )
            if self.mean_penalty_weight != 0:
                raise ValueError(
                    "separated-variance optimizers do not support a mean penalty"
                )
            if self.joint_polish_sweeps != 0 or self.joint_polish_max_nfev is not None:
                raise ValueError(
                    "separated-variance optimizers do not support joint polish"
                )
        q_bounds = np.asarray(self.variance_formula_q_bounds, dtype=float)
        if (
            q_bounds.shape != (2,)
            or not np.all(np.isfinite(q_bounds))
            or not 0 < q_bounds[0] < q_bounds[1] < 1
        ):
            raise ValueError(
                "variance_formula_q_bounds must satisfy 0 < low < high < 1"
            )
        if (
            not np.isfinite(self.experimental_uniform_center_fraction)
            or not 0 < self.experimental_uniform_center_fraction < 1
        ):
            raise ValueError("experimental_uniform_center_fraction must lie in (0, 1)")
        if not (
            isinstance(self.test_center_noise_std, str)
            and self.test_center_noise_std == "auto"
        ):
            noise_std = np.asarray(self.test_center_noise_std, dtype=float)
            if (
                noise_std.ndim > 1
                or not np.all(np.isfinite(noise_std))
                or np.any(noise_std < 0)
            ):
                raise ValueError(
                    "test_center_noise_std must be 'auto', a non-negative scalar, "
                    "or a non-negative feature vector"
                )

    def _set_explicit_initialization(self, init):
        if isinstance(init, str):
            if init != "kmeans":
                raise ValueError("the only string initialization is 'kmeans'")
            return
        if not isinstance(init, Mapping):
            raise TypeError("init must be 'kmeans' or a parameter mapping")
        if "means" not in init or "sigmas" not in init:
            raise ValueError("explicit init requires means and sigmas")

        self.means_ = np.asarray(init["means"], dtype=float).copy()
        self.sigmas_ = np.asarray(init["sigmas"], dtype=float).reshape(-1).copy()
        if "amplitudes" in init and init["amplitudes"] is not None:
            self.amplitudes_ = np.asarray(
                init["amplitudes"],
                dtype=float,
            ).copy()
        if "weights" in init and init["weights"] is not None:
            self._initial_weights = (
                np.asarray(
                    init["weights"],
                    dtype=float,
                )
                .reshape(-1)
                .copy()
            )

    def _ensure_initialized(self, X):
        if self.means_ is None or self.sigmas_ is None:
            initialized = initialize_dimension_free_one_s_gmm(
                data=X,
                n_components=self.k,
                s=1.0,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            self.means_ = initialized["means"]
            self.sigmas_ = initialized["sigmas"]
            self._initial_weights = initialized["weights"]

        self.means_ = np.asarray(self.means_, dtype=float)
        self.sigmas_ = np.asarray(self.sigmas_, dtype=float).reshape(-1)
        if self.means_.shape != (self.k, X.shape[1]):
            raise ValueError("initialized means must have shape (n_components, d)")
        if self.sigmas_.shape != (self.k,) or np.any(self.sigmas_ <= 0):
            raise ValueError("initialized sigmas must be positive and match components")
        if not np.all(np.isfinite(self.means_)) or not np.all(
            np.isfinite(self.sigmas_)
        ):
            raise ValueError("initial parameters must be finite")
        if self._initial_weights is not None:
            if self._initial_weights.shape != (self.k,):
                raise ValueError("initialized weights must match components")
            if np.any(self._initial_weights < 0) or not np.isfinite(
                np.sum(self._initial_weights)
            ):
                raise ValueError("initialized weights must be finite and non-negative")

    @staticmethod
    def _automatic_noise_std(X):
        _, d = X.shape
        unique = np.unique(X, axis=0)
        if unique.shape[0] >= 2:
            distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
            positive = distances[np.isfinite(distances) & (distances > 0)]
            if positive.size:
                return float(np.median(positive) / np.sqrt(d))

        centered = X - np.mean(X, axis=0, keepdims=True)
        fallback = 0.1 * np.sqrt(np.mean(centered * centered))
        if fallback > 0 and np.isfinite(fallback):
            return float(fallback)
        scale = max(1.0, float(np.max(np.abs(X))))
        return float(np.sqrt(np.finfo(float).eps) * scale)

    def _resolve_noise_std(self, X):
        if (
            isinstance(self.test_center_noise_std, str)
            and self.test_center_noise_std == "auto"
        ):
            return self._automatic_noise_std(X)
        noise_std = np.asarray(self.test_center_noise_std, dtype=float)
        if noise_std.ndim == 0:
            return float(noise_std)
        if noise_std.shape != (X.shape[1],):
            raise ValueError("vector test_center_noise_std must have shape (d,)")
        return noise_std.copy()

    def _sample_base_data_centers(self, X, rng, candidate_indices=None):
        if candidate_indices is None:
            candidate_indices = np.arange(X.shape[0])
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        candidates = X[candidate_indices]
        n_candidates = candidates.shape[0]
        n_centers = n_candidates if self.n_test_centers is None else self.n_test_centers
        if self.test_center_mode == "data" and self.n_test_centers is None:
            self.test_center_noise_std_ = 0.0
            self.sampled_data_center_indices_ = candidate_indices.copy()
            return candidates.copy()

        replace = self.test_center_mode == "noisy_data" or n_centers > n_candidates
        local_indices = rng.choice(n_candidates, size=n_centers, replace=replace)
        self.sampled_data_center_indices_ = candidate_indices[local_indices]
        centers = candidates[local_indices].copy()
        if self.test_center_mode == "data":
            self.test_center_noise_std_ = 0.0
            return centers

        noise_std = self._resolve_noise_std(X)
        self.test_center_noise_std_ = noise_std
        centers += rng.normal(size=centers.shape) * noise_std
        return centers

    def _select_data_center_candidates(self, X):
        n_samples = X.shape[0]
        if self.test_center_fraction == 1.0:
            self.test_center_selection_s_ = None
            self.data_center_kernel_counts_ = None
            self.selected_data_center_indices_ = np.arange(n_samples)
            self.selected_data_center_fraction_ = 1.0
            return self.selected_data_center_indices_

        n_selected = max(
            1,
            int(np.ceil(self.test_center_fraction * n_samples)),
        )
        selection_s = float(np.min(np.asarray(self.s_values, dtype=float)))
        counts = compute_kernel_counts(
            data=X,
            test_centers=X,
            s=selection_s,
            leave_out_indices=(np.arange(n_samples) if self.leave_one_out else None),
        )
        ranked_indices = np.argsort(-counts, kind="stable")
        selected_indices = np.sort(ranked_indices[:n_selected])

        self.test_center_selection_s_ = selection_s
        self.data_center_kernel_counts_ = counts
        self.selected_data_center_indices_ = selected_indices
        self.selected_data_center_fraction_ = n_selected / n_samples
        return selected_indices

    def _sample_test_centers_near_means(self, n_samples, rng):
        if self.r_tests_near_means == 0:
            return None
        tests_per_component = int(np.ceil(self.r_tests_near_means * n_samples / self.k))
        sampled = []
        for component in range(self.k):
            noise = rng.normal(size=(tests_per_component, self.means_.shape[1]))
            sampled.append(
                self.means_[component]
                + self.alpha_std_tests * self.sigmas_[component] * noise
            )
        return np.vstack(sampled)

    @staticmethod
    def _random_unit_directions(rng, shape):
        directions = rng.normal(size=shape)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions /= np.maximum(norms, np.finfo(float).tiny)
        return directions

    def _build_test_functions(
        self,
        X,
        rng,
        candidate_indices=None,
        experimental_center_plan=None,
        contamination_mask=None,
    ):
        if experimental_center_plan is None:
            data_centers = self._sample_base_data_centers(
                X,
                rng,
                candidate_indices=candidate_indices,
            )
            if contamination_mask is None:
                data_center_roles = np.zeros(data_centers.shape[0], dtype=np.int8)
            else:
                data_center_roles = label_observed_center_roles(
                    self.sampled_data_center_indices_,
                    contamination_mask,
                )
        else:
            # TODO(research): Isolated oracle override for the negative-space
            # mechanism experiment.  The ordinary sampling branch above remains
            # untouched and this branch should be deleted or replaced afterwards.
            data_centers = np.asarray(
                experimental_center_plan["centers"], dtype=float
            ).copy()
            self.sampled_data_center_indices_ = np.asarray(
                experimental_center_plan["source_indices"], dtype=int
            ).copy()
            data_center_roles = np.asarray(
                experimental_center_plan["roles"], dtype=np.int8
            ).copy()
            if data_centers.shape != X.shape:
                raise ValueError(
                    "experimental oracle center plans must preserve center count"
                )
            if self.sampled_data_center_indices_.shape != (X.shape[0],):
                raise ValueError("experimental source indices have invalid shape")
            if data_center_roles.shape != (X.shape[0],):
                raise ValueError("experimental center roles have invalid shape")
            self.test_center_noise_std_ = 0.0
        self.data_test_centers_ = data_centers
        self.data_test_center_roles_ = data_center_roles
        base_centers = data_centers
        near_means = self._sample_test_centers_near_means(X.shape[0], rng)
        if near_means is not None:
            base_centers = np.vstack([base_centers, near_means])
            base_center_roles = np.concatenate(
                [
                    data_center_roles,
                    np.full(
                        near_means.shape[0],
                        CENTER_ROLE_NEAR_MEAN,
                        dtype=np.int8,
                    ),
                ]
            )
        else:
            base_center_roles = data_center_roles.copy()
        self.base_test_centers_ = base_centers
        self.base_test_center_roles_ = base_center_roles
        if self.leave_one_out:
            base_leave_out_indices = self.sampled_data_center_indices_.copy()
            if near_means is not None:
                base_leave_out_indices = np.concatenate(
                    [
                        base_leave_out_indices,
                        np.full(near_means.shape[0], -1, dtype=int),
                    ]
                )
            self.base_leave_out_indices_ = base_leave_out_indices
        else:
            self.base_leave_out_indices_ = None

        d = X.shape[1]
        blocks = []
        counts = {
            0: self.n_zero_moments,
            1: self.n_first_moments,
            2: self.n_second_moments,
        }
        for order in (0, 1, 2):
            count = counts[order]
            if count == 0:
                continue
            centers = np.tile(base_centers, (count, 1))
            if order == 0:
                directions = np.zeros_like(centers)
                second_directions = None
            else:
                direction_shape = (count, base_centers.shape[0], d)
                directions = self._random_unit_directions(
                    rng,
                    direction_shape,
                ).reshape(-1, d)
                if order == 2 and self.second_moment_mode == "two_directions":
                    second_directions = self._random_unit_directions(
                        rng,
                        direction_shape,
                    ).reshape(-1, d)
                elif order == 2:
                    second_directions = directions.copy()
                else:
                    second_directions = None
            leave_out_indices = (
                None
                if self.base_leave_out_indices_ is None
                else np.tile(self.base_leave_out_indices_, count)
            )
            blocks.append(
                {
                    "moment_order": order,
                    "moment_family": order,
                    "compensated": False,
                    "test_centers": centers,
                    "test_directions": directions,
                    "second_test_directions": second_directions,
                    "leave_out_indices": leave_out_indices,
                }
            )

        if self.n_radial_second_moments:
            centers = base_centers.copy()
            leave_out_indices = (
                None
                if self.base_leave_out_indices_ is None
                else self.base_leave_out_indices_.copy()
            )
            blocks.append(
                {
                    "moment_order": 2,
                    "moment_family": RADIAL_SECOND_MOMENT,
                    "compensated": self.compensated_radial_second_moments,
                    "test_centers": centers,
                    "test_directions": np.zeros_like(centers),
                    "second_test_directions": None,
                    "leave_out_indices": leave_out_indices,
                }
            )

        self.moment_test_functions_ = {
            block["moment_family"]: block for block in blocks
        }
        self.test_centers_ = np.vstack([block["test_centers"] for block in blocks])
        self.test_directions_ = np.vstack(
            [block["test_directions"] for block in blocks]
        )
        self.second_test_directions_ = np.vstack(
            [
                (
                    block["second_test_directions"]
                    if block["second_test_directions"] is not None
                    else np.zeros_like(block["test_centers"])
                )
                for block in blocks
            ]
        )
        self.moment_orders_ = np.concatenate(
            [
                np.full(block["test_centers"].shape[0], block["moment_order"], int)
                for block in blocks
            ]
        )
        self.moment_families_ = np.concatenate(
            [
                np.full(
                    block["test_centers"].shape[0],
                    block["moment_family"],
                    dtype=object,
                )
                for block in blocks
            ]
        )
        return blocks

    def _ensure_s_values(self, X, test_centers=None, leave_out_indices=None):
        if self.s_values is not None:
            s_values = np.asarray(self.s_values, dtype=float).reshape(-1)
            if (
                s_values.size == 0
                or not np.all(np.isfinite(s_values))
                or np.any(s_values <= 0)
            ):
                raise ValueError("s_values must contain finite positive values")
            self.s_values = s_values.tolist()
            return

        if (
            self.target_neighbor_counts is None
            and self.target_neighbor_fractions is None
        ):
            self.s_values = [float(np.sqrt(X.shape[1]))]
            return
        if self.target_neighbor_fractions is None:
            targets = self.target_neighbor_counts
            if isinstance(targets, str) and targets == "auto":
                targets = None
        else:
            maximum_count = float(X.shape[0])
            active_leave_out_indices = (
                self.base_leave_out_indices_
                if test_centers is None
                else leave_out_indices
            )
            if active_leave_out_indices is not None:
                maximum_count -= np.mean(
                    np.asarray(active_leave_out_indices, dtype=int) >= 0
                )
            targets = maximum_count * np.asarray(
                self.target_neighbor_fractions,
                dtype=float,
            )
        self.s_values = select_s_values_by_average_kernel_count(
            X,
            self.k,
            targets,
            self.base_test_centers_ if test_centers is None else test_centers,
            leave_out_indices=(
                self.base_leave_out_indices_
                if test_centers is None
                else leave_out_indices
            ),
        ).tolist()

    @staticmethod
    def _invoke_Z_callback(callback, kwargs):
        signature = inspect.signature(callback)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return callback(**kwargs)
        accepted = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
        return callback(**accepted)

    def _compute_Z(self, X, block, s, Z_callback):
        callback = Z_callback
        if isinstance(callback, Mapping):
            family = block["moment_family"]
            callback = callback.get(family)
            if callback is None and family == block["moment_order"]:
                callback = Z_callback.get(str(block["moment_order"]))
        if callback is None:
            if block["moment_family"] == RADIAL_SECOND_MOMENT:
                return compute_Z_radial_second_moment(
                    data=X,
                    test_centers=block["test_centers"],
                    s=s,
                    compensated=block["compensated"],
                    leave_out_indices=block["leave_out_indices"],
                )
            return compute_Z_moment(
                data=X,
                test_centers=block["test_centers"],
                test_directions=block["test_directions"],
                second_test_directions=block["second_test_directions"],
                s=s,
                moment_order=block["moment_order"],
                leave_out_indices=block["leave_out_indices"],
            )
        kwargs = {
            "X": X,
            "data": X,
            "test_centers": block["test_centers"],
            "test_directions": block["test_directions"],
            "second_test_directions": block["second_test_directions"],
            "s": s,
            "moment_order": block["moment_order"],
            "moment_family": block["moment_family"],
            "compensated": block["compensated"],
            "leave_out_indices": block["leave_out_indices"],
        }
        return self._invoke_Z_callback(callback, kwargs)

    def _observation_blocks_for_s(
        self,
        X,
        test_function_blocks,
        s,
        amplitude_group,
        Z_callback,
    ):
        observations = []
        for block in test_function_blocks:
            observation = dict(block)
            moment_order = block["moment_order"]
            moment_family = block["moment_family"]
            if moment_family == RADIAL_SECOND_MOMENT:
                moment_scale = 1.0
            else:
                moment_scale = (
                    float(s) ** (-moment_order) if self.scale_normalize_moments else 1.0
                )
            observation.update(
                {
                    "Z": self._compute_Z(X, block, s, Z_callback),
                    "s": s,
                    "amplitude_group": amplitude_group,
                    "moment_scale": moment_scale,
                    "loss_weight": self.moment_weights[moment_family],
                    "normalize_loss": self.normalize_moment_losses,
                }
            )
            observations.append(observation)
        return observations

    def _optimization_kwargs(self, means_reference, verbose):
        return {
            "means_init": self.means_,
            "sigmas_init": self.sigmas_,
            "n_outer": self.max_steps,
            "geom_sweeps": self.geom_sweeps,
            "ls_max_nfev": self.ls_max_nfev,
            "active_tol": self.active_tol,
            "eta_step_bound": self.eta_step_bound,
            "m_step_bound": self.m_step_bound,
            "joint_polish_sweeps": self.joint_polish_sweeps,
            "joint_polish_max_nfev": self.joint_polish_max_nfev,
            "means_reference": means_reference,
            "mean_penalty_weight": self.mean_penalty_weight,
            "objective_rtol": self.objective_rtol,
            "objective_atol": self.objective_atol,
            "convergence_patience": self.convergence_patience,
            "amplitude_optimization": self.amplitude_optimization,
            "geometry_optimization": self.geometry_optimization,
            "verbose": verbose,
        }

    def _fit_observation_blocks(
        self,
        observation_blocks,
        *,
        amplitudes_init,
        weights_init,
        means_reference,
        verbose,
        data,
        iteration_callback=None,
    ):
        kwargs = self._optimization_kwargs(means_reference, verbose)
        kwargs.update(
            {
                "amplitudes_init": amplitudes_init,
                "weights_init": weights_init,
                "iteration_callback": iteration_callback,
            }
        )
        if self.optimizer_mode == "legacy":
            return fit_dimension_free_moment_gmm(observation_blocks, **kwargs)
        if self.optimizer_mode == "formula_variance":
            return fit_formula_variance_moment_gmm(
                observation_blocks,
                data=data,
                variance_formula_q_bounds=self.variance_formula_q_bounds,
                **kwargs,
            )
        return fit_joint_variance_moment_gmm(observation_blocks, **kwargs)

    def _amplitudes_from_initial_weights(self, s_values):
        if self._initial_weights is None:
            return self.amplitudes_
        return np.vstack(
            [
                self.get_amplitudes_from_weights(
                    self.means_,
                    self.sigmas_,
                    self._initial_weights,
                    s,
                )
                for s in s_values
            ]
        )

    def fit(
        self,
        X,
        y=None,
        verbose=False,
        Z_callback=None,
        iteration_callback=None,
        d=None,
        n_samples=None,
        contamination_mask=None,
        background_lower=None,
        background_upper=None,
        experimental_test_center_scenario=None,
        experimental_bandwidth_reference="test_centers",
        **kwargs,
    ):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if self.k > X.shape[0] and self.means_ is None:
            raise ValueError("n_components cannot exceed samples for kmeans init")

        self._ensure_initialized(X)
        rng = np.random.default_rng(self.random_state)
        experimental_center_plan = None
        self.experimental_test_center_scenario_ = experimental_test_center_scenario
        if experimental_bandwidth_reference not in (
            "test_centers",
            "all_observed",
        ):
            raise ValueError(
                "experimental_bandwidth_reference must be 'test_centers' or "
                "'all_observed'"
            )
        self.experimental_bandwidth_reference_ = experimental_bandwidth_reference
        if self.experimental_oracle_test_centers:
            # TODO(research): Use a separate RNG stream so oracle center changes
            # do not also change the paired random directions.  This entire path
            # is temporary and must not become a default model feature.
            if experimental_test_center_scenario is None:
                raise ValueError(
                    "experimental_test_center_scenario is required when "
                    "experimental_oracle_test_centers=True"
                )
            if self.test_center_fraction != 1.0:
                raise ValueError(
                    "oracle center scenarios require test_center_fraction=1"
                )
            if self.test_center_mode != "data" or self.n_test_centers is not None:
                raise ValueError(
                    "oracle center scenarios require all data centers before "
                    "their matched replacements"
                )
            seed_sequence = (
                None
                if self.random_state is None
                else np.random.SeedSequence([int(self.random_state), 910_247])
            )
            center_rng = np.random.default_rng(seed_sequence)
            experimental_center_plan = build_experimental_center_plan(
                X,
                scenario=experimental_test_center_scenario,
                contamination_mask=contamination_mask,
                background_lower=background_lower,
                background_upper=background_upper,
                uniform_fraction=self.experimental_uniform_center_fraction,
                rng=center_rng,
            )
        elif experimental_test_center_scenario is not None:
            raise ValueError(
                "dataset supplied experimental_test_center_scenario but the model "
                "did not opt in with experimental_oracle_test_centers=True"
            )
        elif experimental_bandwidth_reference != "test_centers":
            raise ValueError(
                "experimental_bandwidth_reference requires "
                "experimental_oracle_test_centers=True"
            )
        if self.test_center_fraction < 1.0:
            # Density ranking needs a bandwidth.  Select it on all observations
            # first, then use the most local bandwidth to discard isolated
            # data-derived centers.  Near-mean centers are added afterwards and
            # are intentionally never filtered.
            self._ensure_s_values(
                X,
                test_centers=X,
                leave_out_indices=(
                    np.arange(X.shape[0]) if self.leave_one_out else None
                ),
            )
            candidate_indices = self._select_data_center_candidates(X)
            test_function_blocks = self._build_test_functions(
                X,
                rng,
                candidate_indices=candidate_indices,
                experimental_center_plan=experimental_center_plan,
                contamination_mask=contamination_mask,
            )
        else:
            # Keep the original order for exact backward compatibility: build
            # test centers before automatic bandwidth selection.
            candidate_indices = self._select_data_center_candidates(X)
            test_function_blocks = self._build_test_functions(
                X,
                rng,
                candidate_indices=candidate_indices,
                experimental_center_plan=experimental_center_plan,
                contamination_mask=contamination_mask,
            )
            if experimental_bandwidth_reference == "all_observed":
                # TODO(research): Oracle mechanism control: select bandwidths on
                # the original observed-center set even if the fitted probes use
                # clean-only centers.  This isolates center placement from the
                # bandwidth-selection side effect and is not a proposed default.
                self._ensure_s_values(
                    X,
                    test_centers=X,
                    leave_out_indices=(
                        np.arange(X.shape[0]) if self.leave_one_out else None
                    ),
                )
            else:
                self._ensure_s_values(X)

        self.history_per_s = []
        self.optimization_results_ = []
        self.observed_moments_ = {}
        simplex_weights_init = None
        if self.amplitude_optimization == "simplex":
            if self.weights_ is not None:
                simplex_weights_init = self.weights_.copy()
            elif self._initial_weights is not None:
                simplex_weights_init = self._initial_weights.copy()

        progress_iteration = 0

        def callback_for_stage(metadata):
            if iteration_callback is None:
                return None

            def forward(snapshot):
                nonlocal progress_iteration
                iteration_callback(
                    snapshot.with_context(
                        iteration=progress_iteration,
                        metadata=metadata,
                    )
                )
                progress_iteration += 1

            return forward

        if self.s_optimization == "sequential":
            amplitudes_per_s = []
            amplitudes_init = self._amplitudes_from_initial_weights([self.s_values[0]])
            if amplitudes_init is not None and np.asarray(amplitudes_init).ndim == 2:
                amplitudes_init = np.asarray(amplitudes_init)[0]

            for s_index, s in enumerate(self.s_values):
                if verbose:
                    print(f"Fitting moment blocks for s={s:.4f}")
                observation_blocks = self._observation_blocks_for_s(
                    X,
                    test_function_blocks,
                    s,
                    amplitude_group=0,
                    Z_callback=Z_callback,
                )
                for block in observation_blocks:
                    self.observed_moments_[s_index, block["moment_family"]] = (
                        block["moment_scale"] * block["Z"]
                    )
                means_reference = self.means_.copy()
                result = self._fit_observation_blocks(
                    observation_blocks,
                    amplitudes_init=amplitudes_init,
                    weights_init=simplex_weights_init,
                    means_reference=means_reference,
                    verbose=verbose,
                    data=X,
                    iteration_callback=callback_for_stage(
                        {
                            "bandwidth_index": s_index,
                            "bandwidth": float(s),
                            "bandwidth_mode": "sequential",
                        }
                    ),
                )
                self.means_ = result["means"]
                self.sigmas_ = result["sigmas"]
                self.amplitudes_ = result["amplitudes"]
                amplitudes_init = self.amplitudes_
                simplex_weights_init = result["weights"]
                amplitudes_per_s.append(self.amplitudes_.copy())
                self.history_per_s.append(result["history"])
                self.optimization_results_.append(result)

            self.amplitudes_per_s_ = np.vstack(amplitudes_per_s)
            final_result = self.optimization_results_[-1]
            if self.amplitude_optimization == "simplex":
                self.weights_ = final_result["weights"].copy()
                self.amplitudes_per_s_ = np.vstack(
                    [
                        self.get_amplitudes_from_weights(
                            self.means_,
                            self.sigmas_,
                            self.weights_,
                            s,
                        )
                        for s in self.s_values
                    ]
                )
                self.amplitudes_ = self.amplitudes_per_s_[-1].copy()
                self.weights_per_s_ = np.tile(
                    self.weights_[None, :],
                    (len(self.s_values), 1),
                )
            else:
                self.weights_ = self.get_weights_from_amplitudes(
                    self.means_,
                    self.sigmas_,
                    self.amplitudes_,
                    self.s_values[-1],
                )
                self.weights_per_s_ = None
        else:
            observation_blocks = []
            for s_index, s in enumerate(self.s_values):
                blocks_for_s = self._observation_blocks_for_s(
                    X,
                    test_function_blocks,
                    s,
                    amplitude_group=s_index,
                    Z_callback=Z_callback,
                )
                observation_blocks.extend(blocks_for_s)
                for block in blocks_for_s:
                    self.observed_moments_[s_index, block["moment_family"]] = (
                        block["moment_scale"] * block["Z"]
                    )

            means_reference = self.means_.copy()
            amplitudes_init = self._amplitudes_from_initial_weights(self.s_values)
            result = self._fit_observation_blocks(
                observation_blocks,
                amplitudes_init=amplitudes_init,
                weights_init=simplex_weights_init,
                means_reference=means_reference,
                verbose=verbose,
                data=X,
                iteration_callback=callback_for_stage(
                    {
                        "bandwidths": [float(value) for value in self.s_values],
                        "bandwidth_mode": "joint",
                    }
                ),
            )
            self.means_ = result["means"]
            self.sigmas_ = result["sigmas"]
            self.amplitudes_per_s_ = result["amplitudes_per_group"]
            self.amplitudes_ = result["amplitudes"]
            self.history_per_s = [result["history"]]
            self.optimization_results_ = [result]
            if self.amplitude_optimization == "simplex":
                self.weights_ = result["weights"].copy()
                self.weights_per_s_ = np.tile(
                    self.weights_[None, :],
                    (len(self.s_values), 1),
                )
            else:
                self.weights_per_s_ = np.vstack(
                    [
                        self.get_weights_from_amplitudes(
                            self.means_,
                            self.sigmas_,
                            amplitudes,
                            s,
                        )
                        for amplitudes, s in zip(
                            self.amplitudes_per_s_,
                            self.s_values,
                        )
                    ]
                )
                self.weights_ = np.mean(self.weights_per_s_, axis=0)
                self.weights_ /= np.sum(self.weights_)
            final_result = result

        self.objective_ = final_result["objective"]
        self.data_objective_ = final_result["data_objective"]
        self.data_objective_per_order_ = final_result["data_objective_per_order"]
        self.data_objective_per_family_ = final_result["data_objective_per_family"]
        self.penalty_objective_ = final_result["penalty_objective"]
        return self

    @staticmethod
    def get_weights_from_amplitudes(means, sigmas, amplitudes, s):
        means = np.asarray(means, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
        amplitudes = np.asarray(amplitudes, dtype=float).reshape(-1)
        d = means.shape[1]
        weights = amplitudes * np.exp(
            0.5 * d * np.log1p((sigmas * sigmas) / (float(s) ** 2))
        )
        total = np.sum(weights)
        if total <= np.finfo(float).tiny:
            return np.full_like(weights, 1.0 / len(weights), dtype=float)
        return weights / total

    @staticmethod
    def get_amplitudes_from_weights(means, sigmas, weights, s):
        means = np.asarray(means, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        d = means.shape[1]
        return weights * np.exp(
            -0.5 * d * np.log1p((sigmas * sigmas) / (float(s) ** 2))
        )

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": self.sigmas_ * self.sigmas_,
        }
