"""Empirical responses for localized-moment test-function blocks."""

import inspect
from collections.abc import Mapping

from src_np.optimization.utils import RADIAL_SECOND_MOMENT
from src_np.test_functions_response import (
    compute_Z_moment,
    compute_Z_radial_second_moment,
)


def _invoke_callback(callback, kwargs):
    signature = inspect.signature(callback)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callback(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return callback(**accepted)


def compute_block_response(X, block, s, callback):
    if isinstance(callback, Mapping):
        callbacks = callback
        family = block["moment_family"]
        callback = callbacks.get(family)
        if callback is None and family == block["moment_order"]:
            callback = callbacks.get(str(block["moment_order"]))

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
    return _invoke_callback(callback, kwargs)


def observation_blocks_for_s(
    X,
    test_function_blocks,
    s,
    *,
    amplitude_group,
    callback,
    scale_normalize_moments,
    moment_weights,
    normalize_moment_losses,
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
                float(s) ** (-moment_order) if scale_normalize_moments else 1.0
            )
        observation.update(
            {
                "Z": compute_block_response(X, block, s, callback),
                "s": s,
                "amplitude_group": amplitude_group,
                "moment_scale": moment_scale,
                "loss_weight": moment_weights[moment_family],
                "normalize_loss": normalize_moment_losses,
            }
        )
        observations.append(observation)
    return observations
