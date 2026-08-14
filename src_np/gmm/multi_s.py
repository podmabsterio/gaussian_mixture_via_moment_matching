import numpy as np

from src_np.optimization import fit_dimension_free_multi_s_gmm

from .one_s import OneSGaussianMixtureModel


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
