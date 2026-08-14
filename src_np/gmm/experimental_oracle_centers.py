"""Temporary oracle-only center constructions for mechanism experiments.

This module is intentionally isolated from the regular center-selection code so
that it can be deleted once the negative-space-probe hypothesis has been tested.
"""

import numpy as np


CENTER_ROLE_CLEAN = 0
CENTER_ROLE_OBSERVED_OUTLIER = 1
CENTER_ROLE_INDEPENDENT_UNIFORM = 2
CENTER_ROLE_NEAR_MEAN = 3

EXPERIMENTAL_CENTER_SCENARIOS = {
    "standard",
    "oracle_clean_only",
    "oracle_clean_plus_uniform",
    "oracle_clean_plus_outliers",
}


def _validate_mask(mask, n_samples):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_samples,):
        raise ValueError("contamination_mask must have one entry per observation")
    if np.all(mask):
        raise ValueError("oracle center experiments require at least one clean point")
    return mask


def _validate_box(lower, upper, n_features):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != (n_features,) or upper.shape != (n_features,):
        raise ValueError("background bounds must have shape (n_features,)")
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("background bounds must be finite and strictly ordered")
    return lower, upper


def label_observed_center_roles(source_indices, contamination_mask):
    """Label ordinary data-derived centers for response diagnostics."""
    source_indices = np.asarray(source_indices, dtype=int).reshape(-1)
    mask_array = np.asarray(contamination_mask, dtype=bool)
    if mask_array.ndim != 1:
        raise ValueError("contamination_mask must be one-dimensional")
    mask = _validate_mask(mask_array, mask_array.size)
    if np.any(source_indices >= mask.size):
        raise ValueError("source center index exceeds contamination_mask")
    roles = np.full(source_indices.shape, CENTER_ROLE_CLEAN, dtype=np.int8)
    observed = source_indices >= 0
    roles[~observed] = CENTER_ROLE_INDEPENDENT_UNIFORM
    roles[observed & mask[source_indices.clip(min=0)]] = (
        CENTER_ROLE_OBSERVED_OUTLIER
    )
    return roles


def build_experimental_center_plan(
    X,
    *,
    scenario,
    contamination_mask,
    background_lower,
    background_upper,
    uniform_fraction,
    rng,
):
    """Return an equal-size oracle center plan, or ``None`` for standard mode.

    All non-standard plans contain exactly ``len(X)`` centers.  On contaminated
    data, the outlier slots are replaced by clean or independent-uniform probes.
    On clean data, ``uniform_fraction`` clean slots are replaced by uniform
    probes.  Replacement, rather than augmentation, keeps the probe budget fixed.
    """
    # TODO(research): This is deliberately oracle-only mechanism-test code.  If
    # negative-space probes help, replace it with a data-driven proposal that
    # does not use contamination labels or the true background box.
    if scenario not in EXPERIMENTAL_CENTER_SCENARIOS:
        raise ValueError(
            "experimental_test_center_scenario must be one of "
            f"{sorted(EXPERIMENTAL_CENTER_SCENARIOS)}"
        )
    if scenario == "standard":
        return None

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    n_samples, n_features = X.shape
    mask = _validate_mask(contamination_mask, n_samples)

    fraction = float(uniform_fraction)
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("experimental_uniform_center_fraction must lie in (0, 1)")

    centers = X.copy()
    source_indices = np.arange(n_samples, dtype=int)
    roles = np.where(
        mask,
        CENTER_ROLE_OBSERVED_OUTLIER,
        CENTER_ROLE_CLEAN,
    ).astype(np.int8)

    if scenario == "oracle_clean_plus_outliers":
        # TODO(research): This oracle reconstruction should be bitwise identical
        # to standard all-data centers and exists only as an implementation check.
        return {
            "centers": centers,
            "source_indices": source_indices,
            "roles": roles,
        }

    if scenario == "oracle_clean_only":
        # TODO(research): Repeated clean centers are an equal-budget oracle
        # control, not a proposed practical center-selection algorithm.
        slots = np.flatnonzero(mask)
        if slots.size:
            clean_indices = np.flatnonzero(~mask)
            replacements = rng.choice(clean_indices, size=slots.size, replace=True)
            centers[slots] = X[replacements]
            source_indices[slots] = replacements
            roles[slots] = CENTER_ROLE_CLEAN
        return {
            "centers": centers,
            "source_indices": source_indices,
            "roles": roles,
        }

    # TODO(research): The true uniform-background box is oracle information.
    # It is used only to establish whether empty spatial probes can cause the
    # observed robustness; a successful result needs a non-oracle replacement.
    lower, upper = _validate_box(background_lower, background_upper, n_features)
    slots = np.flatnonzero(mask)
    if slots.size == 0:
        n_uniform = max(1, int(round(fraction * n_samples)))
        slots = np.sort(rng.choice(n_samples, size=n_uniform, replace=False))
    centers[slots] = rng.uniform(lower, upper, size=(slots.size, n_features))
    source_indices[slots] = -1
    roles[slots] = CENTER_ROLE_INDEPENDENT_UNIFORM
    return {
        "centers": centers,
        "source_indices": source_indices,
        "roles": roles,
    }
