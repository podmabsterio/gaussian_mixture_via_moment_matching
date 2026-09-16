"""Construction of localized-moment probes and their bandwidths."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_np.gmm.experimental_oracle_centers import (
    CENTER_ROLE_NEAR_MEAN,
    label_observed_center_roles,
)
from src_np.optimization.utils import RADIAL_SECOND_MOMENT
from src_np.test_functions_response import compute_kernel_counts


@dataclass(frozen=True)
class TestFunctionConfig:
    n_components: int
    n_zero_moments: int
    n_first_moments: int
    n_second_moments: int
    n_radial_second_moments: int
    second_moment_mode: str
    compensated_radial_second_moments: bool
    test_center_mode: str
    n_test_centers: int | None
    test_center_noise_std: object
    leave_one_out: bool
    r_tests_near_means: float
    alpha_std_tests: float


@dataclass(frozen=True)
class CandidateSelection:
    indices: np.ndarray
    selection_s: float | None
    kernel_counts: np.ndarray | None
    selected_fraction: float


@dataclass(frozen=True)
class TestFunctionPlan:
    blocks: list[dict]
    sampled_data_center_indices: np.ndarray
    test_center_noise_std: object
    data_test_centers: np.ndarray
    data_test_center_roles: np.ndarray
    base_test_centers: np.ndarray
    base_test_center_roles: np.ndarray
    base_leave_out_indices: np.ndarray | None
    moment_test_functions: dict
    test_centers: np.ndarray
    test_directions: np.ndarray
    second_test_directions: np.ndarray
    moment_orders: np.ndarray
    moment_families: np.ndarray


def resolve_s_values(
    X,
    *,
    n_components,
    s_values,
    target_neighbor_counts,
    target_neighbor_fractions,
    test_centers,
    leave_out_indices,
):
    """Resolve explicit, default, or neighbor-count-derived bandwidths."""
    if s_values is not None:
        resolved = np.asarray(s_values, dtype=float).reshape(-1)
        if (
            resolved.size == 0
            or not np.all(np.isfinite(resolved))
            or np.any(resolved <= 0)
        ):
            raise ValueError("s_values must contain finite positive values")
        return resolved.tolist()

    if target_neighbor_counts is None and target_neighbor_fractions is None:
        return [float(np.sqrt(X.shape[1]))]

    if target_neighbor_fractions is None:
        targets = target_neighbor_counts
        if isinstance(targets, str) and targets == "auto":
            targets = None
    else:
        maximum_count = float(X.shape[0])
        if leave_out_indices is not None:
            maximum_count -= np.mean(np.asarray(leave_out_indices, dtype=int) >= 0)
        targets = maximum_count * np.asarray(
            target_neighbor_fractions,
            dtype=float,
        )

    return select_s_values_by_average_kernel_count(
        X,
        n_components,
        targets,
        test_centers,
        leave_out_indices=leave_out_indices,
    ).tolist()


def select_data_center_candidates(
    X,
    *,
    test_center_fraction,
    s_values,
    leave_one_out,
):
    n_samples = X.shape[0]
    if test_center_fraction == 1.0:
        return CandidateSelection(
            indices=np.arange(n_samples),
            selection_s=None,
            kernel_counts=None,
            selected_fraction=1.0,
        )

    n_selected = max(1, int(np.ceil(test_center_fraction * n_samples)))
    selection_s = float(np.min(np.asarray(s_values, dtype=float)))
    counts = compute_kernel_counts(
        data=X,
        test_centers=X,
        s=selection_s,
        leave_out_indices=(np.arange(n_samples) if leave_one_out else None),
    )
    ranked_indices = np.argsort(-counts, kind="stable")
    selected_indices = np.sort(ranked_indices[:n_selected])
    return CandidateSelection(
        indices=selected_indices,
        selection_s=selection_s,
        kernel_counts=counts,
        selected_fraction=n_selected / n_samples,
    )


def _automatic_noise_std(X):
    _, dimension = X.shape
    unique = np.unique(X, axis=0)
    if unique.shape[0] >= 2:
        distances = cKDTree(unique).query(unique, k=2)[0][:, 1]
        positive = distances[np.isfinite(distances) & (distances > 0)]
        if positive.size:
            return float(np.median(positive) / np.sqrt(dimension))

    centered = X - np.mean(X, axis=0, keepdims=True)
    fallback = 0.1 * np.sqrt(np.mean(centered * centered))
    if fallback > 0 and np.isfinite(fallback):
        return float(fallback)
    scale = max(1.0, float(np.max(np.abs(X))))
    return float(np.sqrt(np.finfo(float).eps) * scale)


def _resolve_noise_std(X, configured_noise_std):
    if isinstance(configured_noise_std, str) and configured_noise_std == "auto":
        return _automatic_noise_std(X)
    noise_std = np.asarray(configured_noise_std, dtype=float)
    if noise_std.ndim == 0:
        return float(noise_std)
    if noise_std.shape != (X.shape[1],):
        raise ValueError("vector test_center_noise_std must have shape (d,)")
    return noise_std.copy()


def _sample_base_data_centers(X, rng, candidate_indices, config):
    candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    candidates = X[candidate_indices]
    n_candidates = candidates.shape[0]
    n_centers = n_candidates if config.n_test_centers is None else config.n_test_centers
    if config.test_center_mode == "data" and config.n_test_centers is None:
        return candidates.copy(), candidate_indices.copy(), 0.0

    replace = config.test_center_mode == "noisy_data" or n_centers > n_candidates
    local_indices = rng.choice(n_candidates, size=n_centers, replace=replace)
    sampled_indices = candidate_indices[local_indices]
    centers = candidates[local_indices].copy()
    if config.test_center_mode == "data":
        return centers, sampled_indices, 0.0

    noise_std = _resolve_noise_std(X, config.test_center_noise_std)
    centers += rng.normal(size=centers.shape) * noise_std
    return centers, sampled_indices, noise_std


def _sample_test_centers_near_means(
    means,
    sigmas,
    n_samples,
    rng,
    config,
):
    if config.r_tests_near_means == 0:
        return None
    tests_per_component = int(
        np.ceil(config.r_tests_near_means * n_samples / config.n_components)
    )
    sampled = []
    for component in range(config.n_components):
        noise = rng.normal(size=(tests_per_component, means.shape[1]))
        sampled.append(
            means[component] + config.alpha_std_tests * sigmas[component] * noise
        )
    return np.vstack(sampled)


def _random_unit_directions(rng, shape):
    directions = rng.normal(size=shape)
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions /= np.maximum(norms, np.finfo(float).tiny)
    return directions


def build_test_function_plan(
    X,
    *,
    means,
    sigmas,
    rng,
    candidate_indices,
    config,
    experimental_center_plan=None,
    contamination_mask=None,
):
    """Build all centers and directional/radial moment probe blocks."""
    if experimental_center_plan is None:
        (
            data_centers,
            sampled_data_center_indices,
            test_center_noise_std,
        ) = _sample_base_data_centers(X, rng, candidate_indices, config)
        if contamination_mask is None:
            data_center_roles = np.zeros(data_centers.shape[0], dtype=np.int8)
        else:
            data_center_roles = label_observed_center_roles(
                sampled_data_center_indices,
                contamination_mask,
            )
    else:
        data_centers = np.asarray(
            experimental_center_plan["centers"], dtype=float
        ).copy()
        sampled_data_center_indices = np.asarray(
            experimental_center_plan["source_indices"], dtype=int
        ).copy()
        data_center_roles = np.asarray(
            experimental_center_plan["roles"], dtype=np.int8
        ).copy()
        if data_centers.shape != X.shape:
            raise ValueError(
                "experimental oracle center plans must preserve center count"
            )
        if sampled_data_center_indices.shape != (X.shape[0],):
            raise ValueError("experimental source indices have invalid shape")
        if data_center_roles.shape != (X.shape[0],):
            raise ValueError("experimental center roles have invalid shape")
        test_center_noise_std = 0.0

    base_centers = data_centers
    near_means = _sample_test_centers_near_means(
        means,
        sigmas,
        X.shape[0],
        rng,
        config,
    )
    if near_means is not None:
        base_centers = np.vstack([base_centers, near_means])
        base_center_roles = np.concatenate(
            [
                data_center_roles,
                np.full(
                    near_means.shape[0],
                    CENTER_ROLE_NEAR_MEAN,
                    dtype=np.int8,
                ),
            ]
        )
    else:
        base_center_roles = data_center_roles.copy()

    if config.leave_one_out:
        base_leave_out_indices = sampled_data_center_indices.copy()
        if near_means is not None:
            base_leave_out_indices = np.concatenate(
                [
                    base_leave_out_indices,
                    np.full(near_means.shape[0], -1, dtype=int),
                ]
            )
    else:
        base_leave_out_indices = None

    dimension = X.shape[1]
    blocks = []
    counts = {
        0: config.n_zero_moments,
        1: config.n_first_moments,
        2: config.n_second_moments,
    }
    for order in (0, 1, 2):
        count = counts[order]
        if count == 0:
            continue
        centers = np.tile(base_centers, (count, 1))
        if order == 0:
            directions = np.zeros_like(centers)
            second_directions = None
        else:
            direction_shape = (count, base_centers.shape[0], dimension)
            directions = _random_unit_directions(rng, direction_shape).reshape(
                -1, dimension
            )
            if order == 2 and config.second_moment_mode == "two_directions":
                second_directions = _random_unit_directions(
                    rng, direction_shape
                ).reshape(-1, dimension)
            elif order == 2:
                second_directions = directions.copy()
            else:
                second_directions = None
        leave_out_indices = (
            None
            if base_leave_out_indices is None
            else np.tile(base_leave_out_indices, count)
        )
        blocks.append(
            {
                "moment_order": order,
                "moment_family": order,
                "compensated": False,
                "test_centers": centers,
                "test_directions": directions,
                "second_test_directions": second_directions,
                "leave_out_indices": leave_out_indices,
            }
        )

    if config.n_radial_second_moments:
        centers = base_centers.copy()
        leave_out_indices = (
            None if base_leave_out_indices is None else base_leave_out_indices.copy()
        )
        blocks.append(
            {
                "moment_order": 2,
                "moment_family": RADIAL_SECOND_MOMENT,
                "compensated": config.compensated_radial_second_moments,
                "test_centers": centers,
                "test_directions": np.zeros_like(centers),
                "second_test_directions": None,
                "leave_out_indices": leave_out_indices,
            }
        )

    moment_test_functions = {block["moment_family"]: block for block in blocks}
    test_centers = np.vstack([block["test_centers"] for block in blocks])
    test_directions = np.vstack([block["test_directions"] for block in blocks])
    second_test_directions = np.vstack(
        [
            (
                block["second_test_directions"]
                if block["second_test_directions"] is not None
                else np.zeros_like(block["test_centers"])
            )
            for block in blocks
        ]
    )
    moment_orders = np.concatenate(
        [
            np.full(block["test_centers"].shape[0], block["moment_order"], int)
            for block in blocks
        ]
    )
    moment_families = np.concatenate(
        [
            np.full(
                block["test_centers"].shape[0],
                block["moment_family"],
                dtype=object,
            )
            for block in blocks
        ]
    )

    return TestFunctionPlan(
        blocks=blocks,
        sampled_data_center_indices=sampled_data_center_indices,
        test_center_noise_std=test_center_noise_std,
        data_test_centers=data_centers,
        data_test_center_roles=data_center_roles,
        base_test_centers=base_centers,
        base_test_center_roles=base_center_roles,
        base_leave_out_indices=base_leave_out_indices,
        moment_test_functions=moment_test_functions,
        test_centers=test_centers,
        test_directions=test_directions,
        second_test_directions=second_test_directions,
        moment_orders=moment_orders,
        moment_families=moment_families,
    )
