from src_np.alternating_optimization import (
    fit_dimension_free_one_s_gmm,
    fit_dimension_free_one_s_gmm_joint,
    fit_dimension_free_multi_s_gmm,
)
from src_np.test_functions_response import compute_Z_one_s
from src_np.kmeans_initialization import initialize_dimension_free_one_s_gmm
from src_np.bandwidth_selection import select_s_values_by_average_kernel_count

import numpy as np


class OneSGaussianMixtureModel:
    def __init__(
        self,
        n_components,
        s_values=None,
        target_neighbor_counts=None,
        init="kmeans",
        n_init=1,
        mean_penalty_weight=0.0,
        objective_rtol=1e-6,
        objective_atol=0.0,
        convergence_patience=2,
        num_directions=1,
        include_base_kernel=False,
        r_tests_near_means=0.0,
        alpha_std_tests=0.1,
        joint_optimization=False,
        random_state=None,
        max_steps=50,
    ):
        self.k = n_components
        self.s_values = s_values
        self.target_neighbor_counts = target_neighbor_counts
        self.n_init = n_init
        self.mean_penalty_weight = mean_penalty_weight
        self.objective_rtol = objective_rtol
        self.objective_atol = objective_atol
        self.convergence_patience = convergence_patience
        self.means_ = None
        self.sigmas_ = None
        self.amplitudes_ = None
        self.num_directions = int(num_directions)
        self.include_base_kernel = bool(include_base_kernel)
        self.r_tests_near_means = float(r_tests_near_means)
        self.alpha_std_tests = float(alpha_std_tests)
        self.joint_optimization = bool(joint_optimization)
        self.random_state = random_state
        self.max_steps = int(max_steps)

        if n_init > 1 and init != "kmeans":
            raise ValueError(
                "n_init > 1 is only supported with 'kmeans' initialization."
            )
        if self.num_directions < 1:
            raise ValueError("num_directions must be positive")
        if self.r_tests_near_means < 0:
            raise ValueError("r_tests_near_means must be non-negative")
        if self.alpha_std_tests < 0:
            raise ValueError("alpha_std_tests must be non-negative")

        if isinstance(init, dict):
            self.means_ = init.get("means")
            self.sigmas_ = init.get("sigmas")
            weights = init.get("weights", None)
            if weights is not None:
                self.amplitudes_ = self.get_amplitudes_from_weights(
                    self.means_, self.sigmas_, weights, s_values[0] if s_values else 1.0
                )
            self.amplitudes_ = init.get("amplitudes", None)

    def _sample_test_centers_near_means(self, n_samples, rng):
        if self.r_tests_near_means == 0:
            return None

        tests_per_component = int(np.ceil(self.r_tests_near_means * n_samples / self.k))
        if tests_per_component < 1:
            return None

        sampled_centers = []
        for k in range(self.k):
            noise = rng.normal(size=(tests_per_component, self.means_.shape[1]))
            centers_k = (
                self.means_[k]
                + self.alpha_std_tests * self.sigmas_[k] * noise
            )
            sampled_centers.append(centers_k)

        return np.vstack(sampled_centers)

    def _ensure_initialized(self, X):
        if self.means_ is not None and self.sigmas_ is not None:
            return

        init_result = initialize_dimension_free_one_s_gmm(
            data=X,
            n_components=self.k,
            s=1.0,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.means_ = init_result["means"]
        self.sigmas_ = init_result["sigmas"]
        self.amplitudes_ = init_result.get("amplitudes")

    def _build_test_functions(self, X, rng):
        n_samples, d = X.shape
        mean_test_centers = self._sample_test_centers_near_means(n_samples, rng)
        if mean_test_centers is None:
            base_test_centers = X
        else:
            base_test_centers = np.vstack([X, mean_test_centers])

        self.base_test_centers_ = base_test_centers

        # Direction-major layout makes the R=1 probes a prefix of R=2/R=3
        # for the same random_state, which is useful for controlled sweeps.
        directional_test_centers = np.tile(base_test_centers, (self.num_directions, 1))
        directional_directions = rng.normal(
            size=(self.num_directions, base_test_centers.shape[0], d)
        ).reshape(-1, d)
        direction_norms = np.linalg.norm(directional_directions, axis=1, keepdims=True)
        directional_directions /= np.maximum(direction_norms, np.finfo(float).tiny)

        if self.include_base_kernel:
            base_directions = np.zeros_like(base_test_centers)
            test_centers = np.vstack([base_test_centers, directional_test_centers])
            directions = np.vstack([base_directions, directional_directions])
        else:
            test_centers = directional_test_centers
            directions = directional_directions

        self.test_centers_ = test_centers
        self.test_directions_ = directions
        return test_centers, directions

    def _ensure_s_values(self, X, test_centers, d):
        if self.s_values is not None:
            return

        if self.target_neighbor_counts is None:
            self.s_values = [d**0.5]
            return

        target_neighbor_counts = self.target_neighbor_counts
        if (
            isinstance(target_neighbor_counts, str)
            and target_neighbor_counts == "auto"
        ):
            target_neighbor_counts = None
        self.s_values = select_s_values_by_average_kernel_count(
            X,
            self.k,
            target_neighbor_counts,
            test_centers,
        )

    def _compute_Z_for_s(self, X, test_centers, directions, s, Z_callback):
        if Z_callback is not None:
            return Z_callback(
                X=X,
                test_centers=test_centers,
                directions=directions,
                s=s,
            )
        return compute_Z_one_s(X, test_centers, directions, s)

    def _fit_from_init_one_s(
        self,
        Z,
        test_centers,
        test_directions,
        s,
        means_reference,
        verbose=False,
    ):
        fit_function = (
            fit_dimension_free_one_s_gmm_joint
            if self.joint_optimization
            else fit_dimension_free_one_s_gmm
        )
        result = fit_function(
            Z=Z,
            test_centers=test_centers,
            test_directions=test_directions,
            s=s,
            means_init=self.means_,
            sigmas_init=self.sigmas_,
            amplitudes_init=self.amplitudes_,
            means_reference=means_reference,
            mean_penalty_weight=self.mean_penalty_weight,
            objective_rtol=self.objective_rtol,
            objective_atol=self.objective_atol,
            convergence_patience=self.convergence_patience,
            verbose=verbose,
            n_outer=self.max_steps,
        )
        self.means_ = result["means"]
        self.sigmas_ = result["sigmas"]
        self.amplitudes_ = result["amplitudes"]
        return result

    def fit(
        self,
        X,
        y=None,
        verbose=False,
        Z_callback=None,
        d=None,
        n_samples=None,
        **kwargs,
    ):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2
        n_samples, d = X.shape

        self._ensure_initialized(X)
        rng = np.random.default_rng(self.random_state)
        test_centers, directions = self._build_test_functions(X, rng)

        self.history_per_s = []

        self._ensure_s_values(X, test_centers, d)

        for s in self.s_values:
            if verbose:
                print(f"Fitting for s={s:.4f}")
            means_reference = self.means_.copy()

            Z = self._compute_Z_for_s(X, test_centers, directions, s, Z_callback)

            result = self._fit_from_init_one_s(
                Z,
                test_centers,
                directions,
                s,
                means_reference=means_reference,
                verbose=verbose,
            )
            self.history_per_s.append(result["history"])

        self.weights_ = self._get_weights(self.s_values[-1])

    @staticmethod
    def get_weights_from_amplitudes(means, sigmas, amplitudes, s):
        d = means.shape[1]
        weights = amplitudes * np.exp(0.5 * d * np.log1p((sigmas * sigmas) / (s * s)))
        weight_sum = np.sum(weights)
        if weight_sum <= np.finfo(float).tiny:
            return np.full_like(weights, 1.0 / len(weights), dtype=float)
        weights /= weight_sum
        return weights

    @staticmethod
    def get_amplitudes_from_weights(means, sigmas, weights, s):
        d = means.shape[1]
        amplitudes = weights * np.exp(-0.5 * d * np.log1p((sigmas * sigmas) / (s * s)))
        return amplitudes

    def _get_weights(self, s):
        return self.get_weights_from_amplitudes(
            self.means_, self.sigmas_, self.amplitudes_, s
        )

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": self.sigmas_ * self.sigmas_,
        }


class MultiSGaussianMixtureModel(OneSGaussianMixtureModel):
    def _fit_from_init_multi_s(
        self,
        Z_list,
        test_centers,
        test_directions,
        s_values,
        means_reference,
        verbose=False,
    ):
        result = fit_dimension_free_multi_s_gmm(
            Z_list=Z_list,
            test_centers=test_centers,
            test_directions=test_directions,
            s_values=s_values,
            means_init=self.means_,
            sigmas_init=self.sigmas_,
            amplitudes_init=self.amplitudes_,
            means_reference=means_reference,
            mean_penalty_weight=self.mean_penalty_weight,
            objective_rtol=self.objective_rtol,
            objective_atol=self.objective_atol,
            convergence_patience=self.convergence_patience,
            verbose=verbose,
            n_outer=self.max_steps,
        )
        self.means_ = result["means"]
        self.sigmas_ = result["sigmas"]
        self.amplitudes_per_s_ = result["amplitudes_per_s"]
        self.amplitudes_ = result["amplitudes"]
        return result

    def _get_weights_multi_s(self):
        weights_per_s = np.vstack(
            [
                self.get_weights_from_amplitudes(
                    self.means_,
                    self.sigmas_,
                    amplitudes,
                    s,
                )
                for amplitudes, s in zip(self.amplitudes_per_s_, self.s_values)
            ]
        )
        weights = np.mean(weights_per_s, axis=0)
        weights /= np.sum(weights)
        self.weights_per_s_ = weights_per_s
        return weights

    def fit(
        self,
        X,
        y=None,
        verbose=False,
        Z_callback=None,
        d=None,
        n_samples=None,
        **kwargs,
    ):
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2
        _, d = X.shape

        self._ensure_initialized(X)
        rng = np.random.default_rng(self.random_state)
        test_centers, directions = self._build_test_functions(X, rng)
        self._ensure_s_values(X, test_centers, d)

        if len(self.s_values) < 1:
            raise ValueError("MultiSGaussianMixtureModel requires at least one s value")

        Z_list = [
            self._compute_Z_for_s(X, test_centers, directions, s, Z_callback)
            for s in self.s_values
        ]

        means_reference = self.means_.copy()
        result = self._fit_from_init_multi_s(
            Z_list,
            test_centers,
            directions,
            self.s_values,
            means_reference=means_reference,
            verbose=verbose,
        )
        self.history_per_s = [result["history"]]
        self.weights_ = self._get_weights_multi_s()
        return self
