"""Mean-only component steps followed by explicit variance formula updates."""

from .separated_variance_common import (
    FORMULA_VARIANCE,
    _fit_separated_variance_moment_gmm,
    variance_from_corrected_radial_moments,
)


def fit_formula_variance_moment_gmm(moment_blocks, means_init, sigmas_init, **kwargs):
    """Fit means component-wise and update each variance using equation (26)."""
    return _fit_separated_variance_moment_gmm(
        moment_blocks,
        means_init,
        sigmas_init,
        variance_mode=FORMULA_VARIANCE,
        **kwargs,
    )


__all__ = [
    "fit_formula_variance_moment_gmm",
    "variance_from_corrected_radial_moments",
]
