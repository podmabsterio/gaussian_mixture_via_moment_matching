"""Optimization orchestration for the public moment GMM estimator."""

import numpy as np

from src_np.optimization import (
    fit_dimension_free_moment_gmm,
    fit_formula_variance_moment_gmm,
    fit_joint_variance_moment_gmm,
)

from .observations import observation_blocks_for_s
from .parameters import amplitudes_from_weights, weights_from_amplitudes


def _optimization_kwargs(model, means_reference, verbose):
    return {
        "means_init": model.means_,
        "sigmas_init": model.sigmas_,
        "n_outer": model.max_steps,
        "geom_sweeps": model.geom_sweeps,
        "ls_max_nfev": model.ls_max_nfev,
        "active_tol": model.active_tol,
        "eta_step_bound": model.eta_step_bound,
        "m_step_bound": model.m_step_bound,
        "joint_polish_sweeps": model.joint_polish_sweeps,
        "joint_polish_max_nfev": model.joint_polish_max_nfev,
        "means_reference": means_reference,
        "mean_penalty_weight": model.mean_penalty_weight,
        "objective_rtol": model.objective_rtol,
        "objective_atol": model.objective_atol,
        "convergence_patience": model.convergence_patience,
        "amplitude_optimization": model.amplitude_optimization,
        "geometry_optimization": model.geometry_optimization,
        "weight_entropy_regularization": model.weight_entropy_regularization,
        "weight_gini_regularization": model.weight_gini_regularization,
        "weight_regularization_schedule": model.weight_regularization_schedule,
        "weight_active_set": model.weight_active_set,
        "weight_solver_random_state": model.random_state,
        "verbose": verbose,
    }


def _fit_observation_blocks(
    model,
    observation_blocks,
    *,
    amplitudes_init,
    weights_init,
    means_reference,
    verbose,
    data,
    iteration_callback,
):
    kwargs = _optimization_kwargs(model, means_reference, verbose)
    kwargs.update(
        {
            "amplitudes_init": amplitudes_init,
            "weights_init": weights_init,
            "iteration_callback": iteration_callback,
        }
    )
    if model.optimizer_mode == "legacy":
        return fit_dimension_free_moment_gmm(observation_blocks, **kwargs)
    if model.optimizer_mode == "formula_variance":
        return fit_formula_variance_moment_gmm(
            observation_blocks,
            data=data,
            variance_formula_q_bounds=model.variance_formula_q_bounds,
            **kwargs,
        )
    return fit_joint_variance_moment_gmm(observation_blocks, **kwargs)


def _amplitudes_from_initial_weights(model, s_values):
    if model._initial_weights is None:
        return model.amplitudes_
    return np.vstack(
        [
            amplitudes_from_weights(
                model.means_,
                model.sigmas_,
                model._initial_weights,
                s,
            )
            for s in s_values
        ]
    )


def _observation_blocks(model, X, test_function_blocks, s, group, callback):
    return observation_blocks_for_s(
        X,
        test_function_blocks,
        s,
        amplitude_group=group,
        callback=callback,
        scale_normalize_moments=model.scale_normalize_moments,
        moment_weights=model.moment_weights,
        normalize_moment_losses=model.normalize_moment_losses,
    )


def _record_observed_moments(model, s_index, blocks):
    for block in blocks:
        model.observed_moments_[s_index, block["moment_family"]] = (
            block["moment_scale"] * block["Z"]
        )


def _run_sequential(
    model,
    X,
    test_function_blocks,
    *,
    verbose,
    Z_callback,
    simplex_weights_init,
    callback_for_stage,
):
    amplitudes_per_s = []
    amplitudes_init = _amplitudes_from_initial_weights(model, [model.s_values[0]])
    if amplitudes_init is not None and np.asarray(amplitudes_init).ndim == 2:
        amplitudes_init = np.asarray(amplitudes_init)[0]

    for s_index, s in enumerate(model.s_values):
        if verbose:
            print(f"Fitting moment blocks for s={s:.4f}")
        observation_blocks = _observation_blocks(
            model,
            X,
            test_function_blocks,
            s,
            0,
            Z_callback,
        )
        _record_observed_moments(model, s_index, observation_blocks)
        means_reference = model.means_.copy()
        result = _fit_observation_blocks(
            model,
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
        model.means_ = result["means"]
        model.sigmas_ = result["sigmas"]
        model.amplitudes_ = result["amplitudes"]
        amplitudes_init = model.amplitudes_
        simplex_weights_init = result["weights"]
        amplitudes_per_s.append(model.amplitudes_.copy())
        model.history_per_s.append(result["history"])
        model.optimization_results_.append(result)

    model.amplitudes_per_s_ = np.vstack(amplitudes_per_s)
    final_result = model.optimization_results_[-1]
    if model.amplitude_optimization == "simplex":
        model.weights_ = final_result["weights"].copy()
        model.amplitudes_per_s_ = np.vstack(
            [
                amplitudes_from_weights(
                    model.means_,
                    model.sigmas_,
                    model.weights_,
                    s,
                )
                for s in model.s_values
            ]
        )
        model.amplitudes_ = model.amplitudes_per_s_[-1].copy()
        model.weights_per_s_ = np.tile(
            model.weights_[None, :],
            (len(model.s_values), 1),
        )
    else:
        model.weights_ = weights_from_amplitudes(
            model.means_,
            model.sigmas_,
            model.amplitudes_,
            model.s_values[-1],
        )
        model.weights_per_s_ = None
    return final_result


def _run_joint(
    model,
    X,
    test_function_blocks,
    *,
    verbose,
    Z_callback,
    simplex_weights_init,
    callback_for_stage,
):
    observation_blocks = []
    for s_index, s in enumerate(model.s_values):
        blocks_for_s = _observation_blocks(
            model,
            X,
            test_function_blocks,
            s,
            s_index,
            Z_callback,
        )
        observation_blocks.extend(blocks_for_s)
        _record_observed_moments(model, s_index, blocks_for_s)

    means_reference = model.means_.copy()
    amplitudes_init = _amplitudes_from_initial_weights(model, model.s_values)
    result = _fit_observation_blocks(
        model,
        observation_blocks,
        amplitudes_init=amplitudes_init,
        weights_init=simplex_weights_init,
        means_reference=means_reference,
        verbose=verbose,
        data=X,
        iteration_callback=callback_for_stage(
            {
                "bandwidths": [float(value) for value in model.s_values],
                "bandwidth_mode": "joint",
            }
        ),
    )
    model.means_ = result["means"]
    model.sigmas_ = result["sigmas"]
    model.amplitudes_per_s_ = result["amplitudes_per_group"]
    model.amplitudes_ = result["amplitudes"]
    model.history_per_s = [result["history"]]
    model.optimization_results_ = [result]
    if model.amplitude_optimization == "simplex":
        model.weights_ = result["weights"].copy()
        model.weights_per_s_ = np.tile(
            model.weights_[None, :],
            (len(model.s_values), 1),
        )
    else:
        model.weights_per_s_ = np.vstack(
            [
                weights_from_amplitudes(
                    model.means_,
                    model.sigmas_,
                    amplitudes,
                    s,
                )
                for amplitudes, s in zip(
                    model.amplitudes_per_s_,
                    model.s_values,
                )
            ]
        )
        model.weights_ = np.mean(model.weights_per_s_, axis=0)
        model.weights_ /= np.sum(model.weights_)
    return result


def run_optimization(
    model,
    X,
    test_function_blocks,
    *,
    verbose,
    Z_callback,
    iteration_callback,
):
    """Fit prepared moment blocks and update the estimator's fitted state."""
    model.history_per_s = []
    model.optimization_results_ = []
    model.observed_moments_ = {}
    simplex_weights_init = None
    if model.amplitude_optimization == "simplex":
        if model.weights_ is not None:
            simplex_weights_init = model.weights_.copy()
        elif model._initial_weights is not None:
            simplex_weights_init = model._initial_weights.copy()

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

    if model.s_optimization == "sequential":
        final_result = _run_sequential(
            model,
            X,
            test_function_blocks,
            verbose=verbose,
            Z_callback=Z_callback,
            simplex_weights_init=simplex_weights_init,
            callback_for_stage=callback_for_stage,
        )
    else:
        final_result = _run_joint(
            model,
            X,
            test_function_blocks,
            verbose=verbose,
            Z_callback=Z_callback,
            simplex_weights_init=simplex_weights_init,
            callback_for_stage=callback_for_stage,
        )

    model.objective_ = final_result["objective"]
    model.data_objective_ = final_result["data_objective"]
    model.data_objective_per_order_ = final_result["data_objective_per_order"]
    model.data_objective_per_family_ = final_result["data_objective_per_family"]
    model.penalty_objective_ = final_result["penalty_objective"]
    model.mean_penalty_objective_ = final_result["mean_penalty_objective"]
    model.weight_regularization_objective_ = final_result[
        "weight_regularization_objective"
    ]
    model.entropy_regularization_objective_ = final_result[
        "entropy_regularization_objective"
    ]
    model.gini_regularization_objective_ = final_result[
        "gini_regularization_objective"
    ]
    model.weight_solver_method_ = final_result["weight_solver_method"]
    model.weight_solver_iterations_ = final_result["weight_solver_iterations"]
    model.weight_solver_converged_ = final_result["weight_solver_converged"]
    model.weight_solver_active_components_ = final_result[
        "weight_solver_active_components"
    ]
    model.weight_solver_active_history_ = final_result[
        "weight_solver_active_history"
    ]
    model.weight_regularization_factor_ = final_result[
        "weight_regularization_factor"
    ]
    model.weight_regularization_factor_history_ = final_result[
        "weight_regularization_factor_history"
    ]
