"""Component-wise mean sweeps followed by one joint variance-only step."""

from .separated_variance_common import (
    JOINT_VARIANCE,
    _fit_separated_variance_moment_gmm,
)


def fit_joint_variance_moment_gmm(moment_blocks, means_init, sigmas_init, **kwargs):
    """Fit all means at fixed variances, then jointly optimize log variances."""
    return _fit_separated_variance_moment_gmm(
        moment_blocks,
        means_init,
        sigmas_init,
        variance_mode=JOINT_VARIANCE,
        **kwargs,
    )


__all__ = ["fit_joint_variance_moment_gmm"]
