import numpy as np


class BaseRunner:
    def __init__(self, gm, use_full_covariance=True):
        self.gm = gm
        self.use_full_covariance = use_full_covariance

    def run(self, X, **kwargs):
        self.gm.fit(X)

    @staticmethod
    def _check_parameters(params_dict, use_full_covariances: bool):
        assert "means" in params_dict, "means not in params"
        assert "weights" in params_dict, "weights not in params"
        assert "covariances" in params_dict, "covariances not in params"
        means, weights, covariances = (
            params_dict["means"],
            params_dict["weights"],
            params_dict["covariances"],
        )
        assert means.ndim == 2, "Means should have shape (n_comp, n_feat)"
        n_comp, n_feat = means.shape

        assert weights.shape == (n_comp,), "Weights should have shape (n_comp,)"

        if use_full_covariances:
            assert covariances.shape == (
                n_comp,
                n_feat,
                n_feat,
            ), "Covariances should have shape (n_comp, n_feat, n_feat)"
        else:
            assert covariances.shape == (
                n_comp,
            ), "Covariances should have shape (n_comp,)"

    @staticmethod
    def _convert_to_full_covariances(covariances, n_features):
        return covariances[:, None, None] * np.eye(n_features)

    def _extract_parameters_not_safe(self):
        raise NotImplementedError

    def extract_parameters(self):
        params = self._extract_parameters_not_safe
        self._check_parameters(params, self.use_full_covariance)
        return params
