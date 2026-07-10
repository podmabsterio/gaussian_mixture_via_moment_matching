import numpy as np


def check_parameters(params_dict, use_full_covariances: bool = True):
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
        assert covariances.shape == (n_comp,), "Covariances should have shape (n_comp,)"


def convert_to_full_covariances(covariances, n_features):
    return covariances[:, None, None] * np.eye(n_features)


def convert_and_check_params(params_dict):
    if params_dict["covariances"].ndim == 1 and params_dict["means"].ndim == 2:
        params_dict["covariances"] = convert_to_full_covariances(
            params_dict["covariances"], params_dict["means"].shape[1]
        )

    check_parameters(params_dict)
    return params_dict
