"""Numerical diagnostics for the moment-GMM optimizer."""

import numpy as np

from experiments.metrics.base_metric import BaseMetric


def _optimization_results(model):
    results = getattr(model, "optimization_results_", None)
    if not results:
        return None
    return results


class OptimizationOuterIterations(BaseMetric):
    """Total number of alternating outer iterations across fitted stages."""

    def __init__(self):
        super().__init__("Optimizer outer iterations")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(
            sum(result.get("n_outer_iter", result["n_iter"]) for result in results)
        )


class OptimizationConverged(BaseMetric):
    """One when every fitted bandwidth stage met its stopping rule."""

    def __init__(self):
        super().__init__("Optimizer converged")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(all(bool(result.get("converged", False)) for result in results))


class GeometryFunctionEvaluations(BaseMetric):
    """Total nonlinear geometry residual evaluations."""

    def __init__(self):
        super().__init__("Geometry nfev")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(sum(result.get("geometry_nfev", 0) for result in results))


class GeometryMaxEvaluationsRate(BaseMetric):
    """Fraction of geometry solves terminated by their evaluation budget."""

    def __init__(self):
        super().__init__("Geometry max-nfev rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        calls = sum(result.get("geometry_calls", 0) for result in results)
        if calls == 0:
            return 0.0
        hits = sum(result.get("geometry_max_nfev_hits", 0) for result in results)
        return float(hits / calls)


class GeometryEtaBoundHitRate(BaseMetric):
    """Fraction of optimized log-variance coordinates ending at a step bound."""

    def __init__(self):
        super().__init__("Eta bound-hit rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        coordinates = sum(
            result.get("geometry_eta_coordinates", 0) for result in results
        )
        if coordinates == 0:
            return 0.0
        hits = sum(result.get("geometry_eta_bound_hits", 0) for result in results)
        return float(hits / coordinates)


class GeometryAcceptanceRate(BaseMetric):
    """Fraction of proposed geometry solves accepted by the objective guard."""

    def __init__(self):
        super().__init__("Geometry acceptance rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        calls = sum(result.get("geometry_calls", 0) for result in results)
        if calls == 0:
            return 0.0
        accepted = sum(result.get("geometry_accepted", 0) for result in results)
        return float(accepted / calls)


class FinalMeanGradientInfinityNorm(BaseMetric):
    """Infinity norm of the final full-objective gradient over all means."""

    def __init__(self):
        super().__init__("Final mean gradient inf")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(results[-1]["final_mean_gradient_inf"])


class FinalLogVarianceGradientInfinityNorm(BaseMetric):
    """Infinity norm of the final full-objective gradient over log variances."""

    def __init__(self):
        super().__init__("Final eta gradient inf")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(results[-1]["final_eta_gradient_inf"])


class JointPolishRelativeGain(BaseMetric):
    """Relative objective decrease produced only by final joint polishing."""

    def __init__(self):
        super().__init__("Joint polish relative gain")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(results[-1].get("joint_polish_relative_gain", 0.0))


class TrueRelativeObjectiveDrop(BaseMetric):
    """Relative drop from the pre-iteration objective to the final objective."""

    def __init__(self):
        super().__init__("True relative objective drop")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        result = results[-1]
        start = float(result["initial_objective"])
        end = float(result["objective"])
        scale = max(abs(start), np.finfo(float).tiny)
        return float((start - end) / scale)


class FormulaVarianceUpdates(BaseMetric):
    """Number of attempted equation-(26) variance updates."""

    def __init__(self):
        super().__init__("Formula variance updates")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(
            sum(result.get("variance_formula_updates", 0) for result in results)
        )


class FormulaVarianceInvalidUpdateRate(BaseMetric):
    """Fraction of formula updates skipped because corrected moments were invalid."""

    def __init__(self):
        super().__init__("Formula invalid-update rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        updates = sum(result.get("variance_formula_updates", 0) for result in results)
        if updates == 0:
            return 0.0
        invalid = sum(
            result.get("variance_formula_invalid_updates", 0) for result in results
        )
        return float(invalid / updates)


class FormulaVarianceQClipRate(BaseMetric):
    """Fraction of valid formula ratios clipped into the configured interval."""

    def __init__(self):
        super().__init__("Formula q-clip rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        valid = sum(
            result.get("variance_formula_updates", 0)
            - result.get("variance_formula_invalid_updates", 0)
            for result in results
        )
        if valid == 0:
            return 0.0
        hits = sum(result.get("variance_formula_q_clip_hits", 0) for result in results)
        return float(hits / valid)


class FormulaVarianceEtaStepBoundHitRate(BaseMetric):
    """Fraction of valid formula updates limited by ``eta_step_bound``."""

    def __init__(self):
        super().__init__("Formula eta-bound rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        valid = sum(
            result.get("variance_formula_updates", 0)
            - result.get("variance_formula_invalid_updates", 0)
            for result in results
        )
        if valid == 0:
            return 0.0
        hits = sum(
            result.get("variance_formula_eta_step_bound_hits", 0) for result in results
        )
        return float(hits / valid)


class JointVarianceSweeps(BaseMetric):
    """Number of joint log-variance-only least-squares sweeps."""

    def __init__(self):
        super().__init__("Joint variance sweeps")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        return float(sum(result.get("variance_joint_calls", 0) for result in results))


def _objective_transitions(results):
    for result in results:
        values = np.r_[float(result["initial_objective"]), result.get("history", [])]
        if values.size >= 2:
            yield values[:-1], values[1:]


class OuterObjectiveIncreaseRate(BaseMetric):
    """Fraction of outer iterations with a numerically meaningful loss increase."""

    def __init__(self):
        super().__init__("Outer objective increase rate")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        increases = 0
        transitions = 0
        for previous, current in _objective_transitions(results):
            scale = np.maximum.reduce(
                [
                    np.abs(previous),
                    np.abs(current),
                    np.full_like(previous, np.finfo(float).tiny),
                ]
            )
            tolerance = 64.0 * np.finfo(float).eps * scale
            increases += int(np.count_nonzero(current - previous > tolerance))
            transitions += previous.size
        return float(increases / transitions) if transitions else 0.0


class MaximumRelativeOuterObjectiveIncrease(BaseMetric):
    """Largest positive relative loss increase over one outer iteration."""

    def __init__(self):
        super().__init__("Max relative outer objective increase")

    def __call__(self, model, **kwargs):
        results = _optimization_results(model)
        if results is None:
            return float("nan")
        maximum = 0.0
        for previous, current in _objective_transitions(results):
            scale = np.maximum(np.abs(previous), np.finfo(float).tiny)
            maximum = max(maximum, float(np.max((current - previous) / scale)))
        return max(maximum, 0.0)
