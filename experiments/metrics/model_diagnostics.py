"""Small diagnostics for interpreting mixture-estimation experiments."""

import numpy as np

from experiments.metrics.base_metric import BaseMetric


def _selected_bandwidths(model):
    values = getattr(model, "s_values", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return values


class MinimumSelectedBandwidth(BaseMetric):
    """Smallest fitted bandwidth, or NaN for models without bandwidths."""

    def __init__(self):
        super().__init__("Min bandwidth")

    def __call__(self, model, **kwargs):
        values = _selected_bandwidths(model)
        return float(np.min(values)) if values is not None else float("nan")


class MaximumSelectedBandwidth(BaseMetric):
    """Largest fitted bandwidth, or NaN for models without bandwidths."""

    def __init__(self):
        super().__init__("Max bandwidth")

    def __call__(self, model, **kwargs):
        values = _selected_bandwidths(model)
        return float(np.max(values)) if values is not None else float("nan")


class MinimumEstimatedWeight(BaseMetric):
    """Smallest fitted component weight, useful for component-stealing checks."""

    def __init__(self):
        super().__init__("Min estimated weight")

    def __call__(self, weights, **kwargs):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size == 0 or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be a non-empty finite vector")
        return float(np.min(weights))


class EstimatedVarianceRatio(BaseMetric):
    """Ratio of largest to smallest average marginal component variance."""

    def __init__(self):
        super().__init__("Estimated variance ratio")

    def __call__(self, covariances, **kwargs):
        covariances = np.asarray(covariances, dtype=float)
        if (
            covariances.ndim != 3
            or covariances.shape[1] != covariances.shape[2]
            or covariances.shape[0] == 0
            or not np.all(np.isfinite(covariances))
        ):
            raise ValueError("covariances must have shape (K, d, d) and be finite")
        variances = np.trace(covariances, axis1=1, axis2=2) / covariances.shape[1]
        if np.any(variances <= 0):
            raise ValueError("component variances must be positive")
        return float(np.max(variances) / np.min(variances))


class SmallBandwidthEmpiricalMomentByCenterRole(BaseMetric):
    """Mean normalized empirical response for one family and center role."""

    def __init__(self, moment_order, center_role, name=None, moment_family=None):
        self.moment_order = int(moment_order)
        self.center_role = int(center_role)
        if self.moment_order not in (0, 1, 2):
            raise ValueError("moment_order must be 0, 1, or 2")
        if moment_family is None:
            self.moment_family = self.moment_order
        elif moment_family == "radial_second" and self.moment_order == 2:
            self.moment_family = moment_family
        else:
            raise ValueError(
                "moment_family must be omitted or be 'radial_second' with "
                "moment_order=2"
            )
        if name is None:
            name = f"Small-s {self.moment_family} role {self.center_role}"
        super().__init__(name)

    def __call__(self, model, **kwargs):
        """Return NaN for baselines or scenarios without the requested role."""
        s_values = _selected_bandwidths(model)
        observed = getattr(model, "observed_moments_", None)
        roles = getattr(model, "base_test_center_roles_", None)
        blocks = getattr(model, "moment_test_functions_", None)
        if s_values is None or observed is None or roles is None or blocks is None:
            return float("nan")
        if self.moment_family not in blocks:
            return float("nan")

        roles = np.asarray(roles, dtype=int).reshape(-1)
        role_mask = roles == self.center_role
        if not np.any(role_mask):
            return float("nan")

        s_index = int(np.argmin(s_values))
        values = np.asarray(observed[s_index, self.moment_family], dtype=float).reshape(
            -1
        )
        n_centers = roles.size
        if values.size % n_centers != 0:
            raise ValueError("moment response count is incompatible with center roles")
        values = values.reshape(-1, n_centers)
        return float(np.mean(values[:, role_mask]))
