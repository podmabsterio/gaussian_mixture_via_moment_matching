import numpy as np

from src_np.test_functions_response import compute_average_kernel_count


def select_s_values_by_average_kernel_count(
    data,
    n_components,
    target_neighbor_counts=None,
    test_centers=None,
    decreasing=True,
    count_rtol=1e-3,
    s_rtol=1e-4,
    max_bisection_steps=40,
    data_batch_size=4096,
    test_batch_size=None,
    max_block_entries=2_000_000,
):
    """Select Gaussian bandwidths by their average soft neighbor counts.

    If targets are omitted, use the three values proposed in ``gmm_last.pdf``:
    ``n / (2K)``, ``n / (sqrt(2) K)``, and ``n / K``.  Bandwidths are returned
    in decreasing order by default, ready for coarse-to-fine continuation.
    """
    data = np.asarray(data, dtype=float)
    if test_centers is None:
        test_centers = data
    else:
        test_centers = np.asarray(test_centers, dtype=float)

    K = int(n_components)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must be a non-empty two-dimensional array")
    if test_centers.ndim != 2 or test_centers.shape[1] != data.shape[1]:
        raise ValueError("test_centers must have shape (J, data.shape[1])")
    if test_centers.shape[0] == 0:
        raise ValueError("test_centers must be non-empty")
    if K < 1:
        raise ValueError("n_components must be positive")
    if count_rtol <= 0 or s_rtol <= 0 or max_bisection_steps < 1:
        raise ValueError("tolerances and max_bisection_steps must be positive")

    n = data.shape[0]
    if target_neighbor_counts is None:
        targets = np.array(
            [n / (2.0 * K), n / (np.sqrt(2.0) * K), n / K],
            dtype=float,
        )
    else:
        targets = np.asarray(target_neighbor_counts, dtype=float).reshape(-1)

    if targets.size == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("target_neighbor_counts must contain finite values")
    if np.any(targets <= 0) or np.any(targets >= n):
        raise ValueError("each target neighbor count must lie strictly between 0 and n")

    centered = data - np.mean(data, axis=0, keepdims=True)
    data_scale = np.sqrt(np.mean(np.sum(centered * centered, axis=1)) / data.shape[1])
    data_scale = max(float(data_scale), np.finfo(float).tiny)

    cache = {}

    def average_count(s):
        s = float(s)
        if s not in cache:
            cache[s] = compute_average_kernel_count(
                data=data,
                test_centers=test_centers,
                s=s,
                data_batch_size=data_batch_size,
                test_batch_size=test_batch_size,
                max_block_entries=max_block_entries,
            )
        return cache[s]

    lower = data_scale * 1e-6
    lower_count = average_count(lower)
    min_target = float(np.min(targets))
    if min_target < lower_count * (1.0 - count_rtol):
        raise ValueError(
            f"smallest target {min_target:.6g} is below the minimum observable "
            f"kernel count {lower_count:.6g} for the supplied centers"
        )

    upper = data_scale
    max_target = float(np.max(targets))
    while average_count(upper) < max_target:
        upper *= 2.0
        if not np.isfinite(upper):
            raise RuntimeError("failed to bracket bandwidths for the requested counts")

    bandwidths = np.empty_like(targets)
    for index, target in enumerate(targets):
        lo = lower
        hi = upper
        for _ in range(max_bisection_steps):
            mid = 0.5 * (lo + hi)
            count = average_count(mid)
            if abs(count - target) <= count_rtol * target:
                break
            if count < target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= s_rtol * max(mid, np.finfo(float).tiny):
                break
        bandwidths[index] = mid

    order = np.argsort(bandwidths)
    if decreasing:
        order = order[::-1]
    return bandwidths[order]
