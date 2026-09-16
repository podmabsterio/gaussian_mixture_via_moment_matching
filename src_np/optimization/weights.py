"""Optimization of mixture weights on the probability simplex."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


WEIGHT_REGULARIZATION_SCHEDULES = frozenset(
    {"constant", "linear", "quadratic", "cosine"}
)


def weight_regularization_schedule_factor(schedule, progress):
    """Return the multiplier for a weight-regularization decay schedule."""
    schedule = str(schedule).lower()
    if schedule not in WEIGHT_REGULARIZATION_SCHEDULES:
        choices = ", ".join(sorted(WEIGHT_REGULARIZATION_SCHEDULES))
        raise ValueError(f"weight regularization schedule must be one of: {choices}")
    progress = float(progress)
    if not np.isfinite(progress):
        raise ValueError("weight regularization schedule progress must be finite")
    progress = float(np.clip(progress, 0.0, 1.0))
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        return 1.0 - progress
    if schedule == "quadratic":
        return 1.0 - progress * progress
    return 0.5 * (1.0 + np.cos(np.pi * progress))


@dataclass(frozen=True)
class SimplexWeightResult:
    weights: np.ndarray
    objective: float
    data_objective: float
    entropy_regularization_objective: float
    gini_regularization_objective: float
    n_iter: int
    converged: bool
    method: str
    n_optimized_components: int

    @property
    def regularization_objective(self):
        return (
            self.entropy_regularization_objective
            + self.gini_regularization_objective
        )


def project_probability_simplex(values):
    """Euclidean projection onto ``{w >= 0, sum(w) = 1}``."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite vector")

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1)
    active = ordered - cumulative / indices > 0
    rho = int(np.flatnonzero(active)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    # Remove the final floating-point normalization error without creating
    # negative coordinates.
    projected /= np.sum(projected)
    return projected


def shannon_entropy(weights):
    """Shannon entropy with the convention ``0 log(0) = 0``."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    positive = weights > 0
    return float(-np.dot(weights[positive], np.log(weights[positive])))


def weight_regularization_objectives(
    weights,
    *,
    entropy_regularization,
    gini_regularization,
):
    """Return entropy and Gini contributions to the minimized objective.

    The Gini term is written as ``-strength * ||w||_2**2``.  On the simplex
    this differs from ``strength * (1 - ||w||_2**2)`` only by a constant.
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    entropy_value = float(entropy_regularization) * shannon_entropy(weights)
    gini_value = -float(gini_regularization) * float(np.dot(weights, weights))
    return entropy_value, gini_value


def _normalized_start(initial_weights, n_components):
    if initial_weights is None:
        return np.full(n_components, 1.0 / n_components, dtype=float)
    start = np.asarray(initial_weights, dtype=float).reshape(-1).copy()
    if start.shape != (n_components,) or not np.all(np.isfinite(start)):
        raise ValueError("initial_weights must be finite and match the design")
    start = np.maximum(start, 0.0)
    total = float(np.sum(start))
    if total <= np.finfo(float).tiny:
        return np.full(n_components, 1.0 / n_components, dtype=float)
    return start / total


def _break_symmetric_start(weights, random_state):
    if weights.size <= 1 or not np.allclose(
        weights,
        weights[0],
        rtol=0.0,
        atol=32.0 * np.finfo(float).eps,
    ):
        return weights
    rng = np.random.default_rng(random_state)
    random_weights = rng.dirichlet(np.ones(weights.size))
    return 0.95 * weights + 0.05 * random_weights


def _objective_parts(
    design,
    target,
    weights,
    entropy_regularization,
    gini_regularization,
):
    residual = design @ weights - target
    data_value = 0.5 * float(np.dot(residual, residual))
    entropy_value, gini_value = weight_regularization_objectives(
        weights,
        entropy_regularization=entropy_regularization,
        gini_regularization=gini_regularization,
    )
    return data_value, entropy_value, gini_value


def _result(
    design,
    target,
    weights,
    entropy_regularization,
    gini_regularization,
    *,
    n_iter,
    converged,
    method,
    n_optimized_components,
):
    data_value, entropy_value, gini_value = _objective_parts(
        design,
        target,
        weights,
        entropy_regularization,
        gini_regularization,
    )
    return SimplexWeightResult(
        weights=np.asarray(weights, dtype=float).copy(),
        objective=data_value + entropy_value + gini_value,
        data_objective=data_value,
        entropy_regularization_objective=entropy_value,
        gini_regularization_objective=gini_value,
        n_iter=int(n_iter),
        converged=bool(converged),
        method=method,
        n_optimized_components=int(n_optimized_components),
    )


def _solve_quadratic_weights(
    design,
    target,
    start,
    *,
    gini_regularization,
    max_iter,
    tolerance,
):
    n_components = design.shape[1]

    def objective(weights):
        residual = design @ weights - target
        return 0.5 * np.dot(residual, residual) - gini_regularization * np.dot(
            weights,
            weights,
        )

    def gradient(weights):
        return design.T @ (design @ weights - target) - 2.0 * (
            gini_regularization * weights
        )

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_components,
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": tolerance, "maxiter": int(max_iter)},
    )
    if not result.success:
        raise RuntimeError(f"simplex weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= np.sum(weights)
    return weights, int(result.nit), bool(result.success)


def _solve_entropy_pgd(
    design,
    target,
    start,
    *,
    entropy_regularization,
    gini_regularization,
    max_iter,
    tolerance,
    momentum,
    entropy_gradient_floor,
):
    """Monotone accelerated projected gradient for the non-convex objective."""
    weights = start.copy()
    previous = weights.copy()
    data_scale = float(np.sum(design * design))
    step = 1.0 / max(data_scale, np.finfo(float).tiny)

    def objective(values):
        return sum(
            _objective_parts(
                design,
                target,
                values,
                entropy_regularization,
                gini_regularization,
            )
        )

    def gradient(values):
        safe = np.maximum(values, entropy_gradient_floor)
        entropy_gradient = -entropy_regularization * (np.log(safe) + 1.0)
        return (
            design.T @ (design @ values - target)
            + entropy_gradient
            - 2.0 * gini_regularization * values
        )

    current_objective = objective(weights)
    converged = False
    n_iter = 0
    for iteration in range(int(max_iter)):
        extrapolated = project_probability_simplex(
            weights + momentum * (weights - previous)
        )
        candidate = project_probability_simplex(
            extrapolated - step * gradient(extrapolated)
        )
        candidate_objective = objective(candidate)

        if candidate_objective > current_objective:
            # Momentum is useful on the long flat directions of the moment
            # design, but restart it whenever it loses monotonicity.
            extrapolated = weights
            local_step = step
            for _ in range(60):
                candidate = project_probability_simplex(
                    weights - local_step * gradient(weights)
                )
                candidate_objective = objective(candidate)
                if candidate_objective <= current_objective:
                    break
                local_step *= 0.5
            else:
                n_iter = iteration
                break
            step = local_step

        delta = float(np.max(np.abs(candidate - weights)))
        objective_delta = abs(current_objective - candidate_objective)
        previous, weights = weights, candidate
        current_objective = candidate_objective
        n_iter = iteration + 1
        step *= 1.02
        if delta <= tolerance and objective_delta <= tolerance * max(
            1.0,
            abs(current_objective),
        ):
            converged = True
            break

    return weights, n_iter, converged


def solve_simplex_weights(
    design,
    target,
    initial_weights=None,
    *,
    entropy_regularization=0.0,
    gini_regularization=0.0,
    random_state=None,
    max_iter=None,
    tolerance=1e-12,
    momentum=0.9,
    entropy_gradient_floor=1e-12,
    use_active_set=True,
    active_tolerance=0.0,
):
    """Solve a moment least-squares problem over mixture weights.

    Non-zero entropy regularization selects momentum PGD by default.  With no
    entropy term, the objective remains quadratic (possibly non-convex under
    Gini regularization) and is handled by the existing SLSQP-style solver.

    If ``use_active_set`` is enabled and ``initial_weights`` are supplied,
    components whose weight is at most ``active_tolerance`` times the largest
    weight are held at zero.  The reduced problem is solved only over the
    remaining design columns and then expanded back to the original component
    count.
    """
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    if design.ndim != 2 or design.shape[0] != target.size or design.shape[1] == 0:
        raise ValueError("design and target shapes do not match")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
        raise ValueError("design and target must contain only finite values")
    entropy_regularization = float(entropy_regularization)
    gini_regularization = float(gini_regularization)
    if (
        not np.isfinite(entropy_regularization)
        or entropy_regularization < 0
        or not np.isfinite(gini_regularization)
        or gini_regularization < 0
    ):
        raise ValueError("weight regularization strengths must be finite and non-negative")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    if tolerance <= 0 or entropy_gradient_floor <= 0:
        raise ValueError("solver tolerances must be positive")
    active_tolerance = float(active_tolerance)
    if not np.isfinite(active_tolerance) or active_tolerance < 0:
        raise ValueError("active_tolerance must be finite and non-negative")

    full_n_components = design.shape[1]
    start = _normalized_start(initial_weights, full_n_components)
    if use_active_set and initial_weights is not None:
        active_threshold = active_tolerance * float(np.max(start))
        active_indices = np.flatnonzero(start > active_threshold)
    else:
        active_indices = np.arange(full_n_components)
    # Large tolerances may otherwise exclude every coordinate.  A simplex
    # problem always needs at least one active component.
    if active_indices.size == 0:
        active_indices = np.array([int(np.argmax(start))])

    active_design = design[:, active_indices]
    active_start = start[active_indices]
    active_start /= np.sum(active_start)
    n_components = active_indices.size
    regularized = entropy_regularization > 0 or gini_regularization > 0
    if regularized:
        active_start = _break_symmetric_start(active_start, random_state)

    if n_components == 1:
        weights = np.zeros(full_n_components, dtype=float)
        weights[active_indices[0]] = 1.0
        return _result(
            design,
            target,
            weights,
            entropy_regularization,
            gini_regularization,
            n_iter=0,
            converged=True,
            method="constant",
            n_optimized_components=1,
        )

    if entropy_regularization > 0:
        iterations = 1000 if max_iter is None else int(max_iter)
        active_weights, n_iter, converged = _solve_entropy_pgd(
            active_design,
            target,
            active_start,
            entropy_regularization=entropy_regularization,
            gini_regularization=gini_regularization,
            max_iter=iterations,
            tolerance=tolerance,
            momentum=momentum,
            entropy_gradient_floor=entropy_gradient_floor,
        )
        method = "momentum_pgd"
    else:
        iterations = max(200, 20 * n_components) if max_iter is None else int(max_iter)
        active_weights, n_iter, converged = _solve_quadratic_weights(
            active_design,
            target,
            active_start,
            gini_regularization=gini_regularization,
            max_iter=iterations,
            tolerance=tolerance,
        )
        method = "quadratic_slsqp"

    weights = np.zeros(full_n_components, dtype=float)
    weights[active_indices] = active_weights
    return _result(
        design,
        target,
        weights,
        entropy_regularization,
        gini_regularization,
        n_iter=n_iter,
        converged=converged,
        method=method,
        n_optimized_components=n_components,
    )


__all__ = [
    "SimplexWeightResult",
    "WEIGHT_REGULARIZATION_SCHEDULES",
    "project_probability_simplex",
    "shannon_entropy",
    "solve_simplex_weights",
    "weight_regularization_schedule_factor",
    "weight_regularization_objectives",
]
