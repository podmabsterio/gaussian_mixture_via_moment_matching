from collections.abc import Mapping

import numpy as np

from src_np.gmm.experimental_oracle_centers import build_experimental_center_plan
from src_np.kmeans_initialization import (
    initialize_data_point_gmm,
    initialize_dimension_free_one_s_gmm,
)
from src_np.optimization.utils import RADIAL_SECOND_MOMENT

from .configuration import (
    canonical_optimizer_mode,
    canonical_s_optimization,
    canonical_second_moment_mode,
    validate_configuration,
)
from .fitting import run_optimization
from .parameters import (
    amplitudes_from_weights,
    mixture_params_dict,
    weights_from_amplitudes,
)
from .probes import (
    TestFunctionConfig,
    build_test_function_plan,
    resolve_s_values,
    select_data_center_candidates,
)


class MomentGaussianMixtureModel:
    """Configurable dimension-free isotropic GMM moment estimator.

    ``n_*_moments`` is the number of probes of that order per base test center.
    Setting a count to zero disables the corresponding order.

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

    ``init='data_points'`` creates one component per observation, independently
    of the constructor's ``n_components`` value, and uses a positive shared
    nearest-neighbor scale for the initial sigmas.
    Positive ``weight_entropy_regularization`` penalizes Shannon entropy and
    switches the simplex weight update to projected gradient with momentum.
    ``weight_gini_regularization`` adds the quadratic
    ``-strength * ||weights||_2**2`` term.  Both regularizers encourage weight
    concentration and require ``amplitude_optimization='simplex'``.
    ``weight_regularization_schedule`` optionally decays both strengths over
    the configured outer optimization steps.
    With ``weight_active_set=True``, a component whose simplex weight is at
    most ``active_tol`` times the largest weight is excluded from subsequent
    weight updates.
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
        weight_entropy_regularization=0.0,
        weight_gini_regularization=0.0,
        weight_regularization_schedule="constant",
        weight_active_set=True,
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
            else tuple(np.asarray(target_neighbor_fractions, dtype=float).reshape(-1))
        )
        self.init = init
        self.n_init = int(n_init)

        self.n_zero_moments = int(n_zero_moments)
        self.n_first_moments = int(n_first_moments)
        self.n_second_moments = int(n_second_moments)
        self.n_radial_second_moments = int(n_radial_second_moments)
        self.compensated_radial_second_moments = bool(compensated_radial_second_moments)
        self.second_moment_mode = canonical_second_moment_mode(second_moment_mode)
        self.moment_weights = {
            0: float(zero_moment_weight),
            1: float(first_moment_weight),
            2: float(second_moment_weight),
            RADIAL_SECOND_MOMENT: float(radial_second_moment_weight),
        }
        self.normalize_moment_losses = bool(normalize_moment_losses)
        self.scale_normalize_moments = bool(scale_normalize_moments)
        self.amplitude_optimization = amplitude_optimization
        self.s_optimization = canonical_s_optimization(s_optimization)
        self.geometry_optimization = geometry_optimization
        self.optimizer_mode = canonical_optimizer_mode(optimizer_mode)
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
        self.weight_entropy_regularization = float(weight_entropy_regularization)
        self.weight_gini_regularization = float(weight_gini_regularization)
        self.weight_regularization_schedule = str(
            weight_regularization_schedule
        ).lower()
        self.weight_active_set = bool(weight_active_set)
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
        validate_configuration(self)
        self._set_explicit_initialization(init)

    def _set_explicit_initialization(self, init):
        if isinstance(init, str):
            if init not in ("kmeans", "data_points"):
                raise ValueError(
                    "string initialization must be 'kmeans' or 'data_points'"
                )
            return
        if not isinstance(init, Mapping):
            raise TypeError(
                "init must be 'kmeans', 'data_points', or a parameter mapping"
            )
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
            if self.init == "data_points":
                # This mode represents the deliberately over-specified model:
                # one initial component for every observation.  In UI runs the
                # constructor may still receive the dataset's oracle K, so the
                # data-dependent component count must be resolved here.
                self.k = X.shape[0]
                initialized = initialize_data_point_gmm(
                    data=X,
                    n_components=self.k,
                    random_state=self.random_state,
                )
                self.data_point_init_indices_ = initialized["indices"].copy()
            else:
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

        self.n_components_ = self.k

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

    def _test_function_config(self):
        return TestFunctionConfig(
            n_components=self.k,
            n_zero_moments=self.n_zero_moments,
            n_first_moments=self.n_first_moments,
            n_second_moments=self.n_second_moments,
            n_radial_second_moments=self.n_radial_second_moments,
            second_moment_mode=self.second_moment_mode,
            compensated_radial_second_moments=(self.compensated_radial_second_moments),
            test_center_mode=self.test_center_mode,
            n_test_centers=self.n_test_centers,
            test_center_noise_std=self.test_center_noise_std,
            leave_one_out=self.leave_one_out,
            r_tests_near_means=self.r_tests_near_means,
            alpha_std_tests=self.alpha_std_tests,
        )

    def _apply_candidate_selection(self, selection):
        self.test_center_selection_s_ = selection.selection_s
        self.data_center_kernel_counts_ = selection.kernel_counts
        self.selected_data_center_indices_ = selection.indices
        self.selected_data_center_fraction_ = selection.selected_fraction

    def _apply_test_function_plan(self, plan):
        self.sampled_data_center_indices_ = plan.sampled_data_center_indices
        self.test_center_noise_std_ = plan.test_center_noise_std
        self.data_test_centers_ = plan.data_test_centers
        self.data_test_center_roles_ = plan.data_test_center_roles
        self.base_test_centers_ = plan.base_test_centers
        self.base_test_center_roles_ = plan.base_test_center_roles
        self.base_leave_out_indices_ = plan.base_leave_out_indices
        self.moment_test_functions_ = plan.moment_test_functions
        self.test_centers_ = plan.test_centers
        self.test_directions_ = plan.test_directions
        self.second_test_directions_ = plan.second_test_directions
        self.moment_orders_ = plan.moment_orders
        self.moment_families_ = plan.moment_families

    def _prepare_experimental_center_plan(
        self,
        X,
        *,
        contamination_mask,
        background_lower,
        background_upper,
        scenario,
        bandwidth_reference,
    ):
        self.experimental_test_center_scenario_ = scenario
        if bandwidth_reference not in ("test_centers", "all_observed"):
            raise ValueError(
                "experimental_bandwidth_reference must be 'test_centers' or "
                "'all_observed'"
            )
        self.experimental_bandwidth_reference_ = bandwidth_reference

        if self.experimental_oracle_test_centers:
            if scenario is None:
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
            return build_experimental_center_plan(
                X,
                scenario=scenario,
                contamination_mask=contamination_mask,
                background_lower=background_lower,
                background_upper=background_upper,
                uniform_fraction=self.experimental_uniform_center_fraction,
                rng=center_rng,
            )
        if scenario is not None:
            raise ValueError(
                "dataset supplied experimental_test_center_scenario but the model "
                "did not opt in with experimental_oracle_test_centers=True"
            )
        if bandwidth_reference != "test_centers":
            raise ValueError(
                "experimental_bandwidth_reference requires "
                "experimental_oracle_test_centers=True"
            )
        return None

    def _resolve_s_values(self, X, test_centers, leave_out_indices):
        self.s_values = resolve_s_values(
            X,
            n_components=self.k,
            s_values=self.s_values,
            target_neighbor_counts=self.target_neighbor_counts,
            target_neighbor_fractions=self.target_neighbor_fractions,
            test_centers=test_centers,
            leave_out_indices=leave_out_indices,
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
        if self.init != "data_points" and self.k > X.shape[0] and self.means_ is None:
            raise ValueError("n_components cannot exceed samples for kmeans init")

        self._ensure_initialized(X)
        rng = np.random.default_rng(self.random_state)
        experimental_center_plan = self._prepare_experimental_center_plan(
            X,
            contamination_mask=contamination_mask,
            background_lower=background_lower,
            background_upper=background_upper,
            scenario=experimental_test_center_scenario,
            bandwidth_reference=experimental_bandwidth_reference,
        )

        leave_out_all = np.arange(X.shape[0]) if self.leave_one_out else None
        if self.test_center_fraction < 1.0:
            # Density ranking requires a bandwidth before centers are filtered.
            self._resolve_s_values(X, X, leave_out_all)

        selection = select_data_center_candidates(
            X,
            test_center_fraction=self.test_center_fraction,
            s_values=self.s_values,
            leave_one_out=self.leave_one_out,
        )
        self._apply_candidate_selection(selection)
        plan = build_test_function_plan(
            X,
            means=self.means_,
            sigmas=self.sigmas_,
            rng=rng,
            candidate_indices=selection.indices,
            config=self._test_function_config(),
            experimental_center_plan=experimental_center_plan,
            contamination_mask=contamination_mask,
        )
        self._apply_test_function_plan(plan)

        if self.test_center_fraction == 1.0:
            if experimental_bandwidth_reference == "all_observed":
                self._resolve_s_values(X, X, leave_out_all)
            else:
                self._resolve_s_values(
                    X,
                    self.base_test_centers_,
                    self.base_leave_out_indices_,
                )

        run_optimization(
            self,
            X,
            plan.blocks,
            verbose=verbose,
            Z_callback=Z_callback,
            iteration_callback=iteration_callback,
        )
        return self

    @staticmethod
    def get_weights_from_amplitudes(means, sigmas, amplitudes, s):
        return weights_from_amplitudes(means, sigmas, amplitudes, s)

    @staticmethod
    def get_amplitudes_from_weights(means, sigmas, weights, s):
        return amplitudes_from_weights(means, sigmas, weights, s)

    def params_dict(self):
        return mixture_params_dict(self.means_, self.sigmas_, self.weights_)
