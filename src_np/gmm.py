from src_np.alternating_optimization import (
    fit_dimension_free_one_s_gmm,
    fit_dimension_free_one_s_gmm_joint,
)
from src_np.test_functions_response import compute_Z_one_s
from src_np.kmeans_initialization import initialize_dimension_free_one_s_gmm

import numpy as np


class OneSGaussianMixtureModel:
    def __init__(
        self,
        k,
        s_values=None,
        point_counts=None,
        init="kmeans",
        n_init=1,
        mean_penalty_weight=0.0,
        objective_rtol=1e-6,
        objective_atol=0.0,
        convergence_patience=2,
        num_directions=1,
        joint_optimization=False,
        random_state=None,
        max_steps=50,
    ):
        self.k = k
        self.s_values = s_values
        self.n_init = n_init
        self.mean_penalty_weight = mean_penalty_weight
        self.objective_rtol = objective_rtol
        self.objective_atol = objective_atol
        self.convergence_patience = convergence_patience
        self.means_ = None
        self.sigmas_ = None
        self.amplitudes_ = None
        self.num_directions = int(num_directions)
        self.joint_optimization = bool(joint_optimization)
        self.random_state = random_state
        self.max_steps = int(max_steps)

        if n_init > 1 and init != "kmeans":
            raise ValueError(
                "n_init > 1 is only supported with 'kmeans' initialization."
            )
        if self.num_directions < 1:
            raise ValueError("num_directions must be positive")

        if isinstance(init, dict):
            self.means_ = init.get("means")
            self.sigmas_ = init.get("sigmas")
            weights = init.get("weights", None)
            if weights is not None:
                self.amplitudes_ = self.get_amplitudes_from_weights(
                    self.means_, self.sigmas_, weights, s_values[0] if s_values else 1.0
                )
            self.amplitudes_ = init.get("amplitudes", None)

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
        if X is not None:
            X = np.asarray(X, dtype=float)
            assert X.ndim == 2
            n_samples, d = X.shape

        if self.means_ is None or self.sigmas_ is None:
            if X is None:
                raise ValueError("X is None and model is not initialized")
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

        if self.s_values is None:
            self.s_values = [d**0.5]

        rng = np.random.default_rng(self.random_state)
        # Direction-major layout makes the R=1 probes a prefix of R=2/R=3
        # for the same random_state, which is useful for controlled sweeps.
        test_centers = np.tile(X, (self.num_directions, 1))
        directions = rng.normal(size=(self.num_directions, n_samples, d)).reshape(-1, d)
        direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.maximum(direction_norms, np.finfo(float).tiny)

        self.history_per_s = {}

        for s in self.s_values:
            if verbose:
                print(f"Fitting for s={s:.4f}")
            means_reference = self.means_.copy()

            if Z_callback is not None:
                Z = Z_callback(
                    X=X, test_centers=test_centers, directions=directions, s=s
                )
            else:
                Z = compute_Z_one_s(X, test_centers, directions, s)

            result = self._fit_from_init_one_s(
                Z,
                test_centers,
                directions,
                s,
                means_reference=means_reference,
                verbose=verbose,
            )
            result["weights"] = self._get_weights(s)
            self.history_per_s[s] = result

        self.weights_ = self._get_weights(self.s_values[-1])

    @staticmethod
    def get_weights_from_amplitudes(means, sigmas, amplitudes, s):
        d = means.shape[1]
        weights = amplitudes * np.exp(0.5 * d * np.log1p((sigmas * sigmas) / (s * s)))
        weights /= np.sum(weights)
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
            "covariances": self.sigmas_,
        }
