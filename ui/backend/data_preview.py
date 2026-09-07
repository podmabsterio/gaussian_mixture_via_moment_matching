from __future__ import annotations

from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.visualization import project_dataset_to_2d

from .declarations import DeclarationError, DeclarationStore


def ellipse_geometry(means: np.ndarray, covariances: np.ndarray) -> list[dict[str, Any]]:
    """Convert projected Gaussian parameters into canvas-ready 1σ ellipses."""
    ellipses = []
    for mean, covariance in zip(means, covariances, strict=True):
        symmetric = (covariance + covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        radii = np.sqrt(eigenvalues)
        ellipses.append(
            {
                "center": [float(mean[0]), float(mean[1])],
                "radii": [float(radii[0]), float(radii[1])],
                "angle": float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
            }
        )
    return ellipses


def display_indices(labels: np.ndarray, seed: int, max_points: int) -> np.ndarray:
    """Choose a deterministic display subset while preserving rare labels."""
    if labels.size <= max_points:
        return np.arange(labels.size)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9_173]))
    required = np.asarray(
        [rng.choice(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    )
    remaining_count = max_points - required.size
    available = np.setdiff1d(np.arange(labels.size), required, assume_unique=True)
    remaining = rng.choice(available, size=remaining_count, replace=False)
    return np.sort(np.concatenate((required, remaining)))


def dataset_labels(dataset: dict[str, Any], n_points: int) -> np.ndarray:
    labels = np.asarray(dataset.get("true_labels", np.zeros(n_points)), dtype=int)
    if labels.shape != (n_points,):
        raise DeclarationError("Generated dataset labels do not match its observations")
    return labels


class DataPreviewService:
    """Generate and project a declared dataset for the Experiment preview."""

    def __init__(self, declarations: DeclarationStore, max_points: int = 3_000):
        self.declarations = declarations
        self.max_points = max_points

    def build(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise DeclarationError("Request body must be an object")
        entry = request.get("dataset")
        if not isinstance(entry, dict):
            raise DeclarationError("dataset must be an object")

        declaration = self.declarations.dataset(entry.get("declaration_id"))
        parameters = self.declarations.resolve_parameters(
            declaration["parameters"],
            entry.get("parameters", {}),
            "dataset.parameters",
        )
        seed = request.get("seed", 1)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647:
            raise DeclarationError("seed must be an integer between 0 and 2147483647")

        try:
            generator = instantiate(
                OmegaConf.create({"_target_": declaration["target"], **parameters})
            )
            dataset = generator.generate(seed)
            projection = project_dataset_to_2d(dataset)
            labels = dataset_labels(dataset, projection["X"].shape[0])
            indices = display_indices(labels, seed, self.max_points)
            ellipses = ellipse_geometry(
                projection["true_means"], projection["true_covariances"]
            )
        except DeclarationError:
            raise
        except Exception as exc:
            raise DeclarationError(f"Could not generate data preview: {exc}") from exc

        points = projection["X"]
        weights = np.asarray(dataset.get("true_weights", []), dtype=float).reshape(-1)
        return {
            "dataset": {
                "declaration_id": declaration["id"],
                "display_name": declaration["display_name"],
            },
            "seed": seed,
            "n_features": int(np.asarray(dataset["X"]).shape[1]),
            "total_points": int(points.shape[0]),
            "displayed_points": int(indices.size),
            "explained_variance_ratio": [
                float(value) for value in projection["explained_variance_ratio"]
            ],
            "points": [
                {
                    "x": float(points[index, 0]),
                    "y": float(points[index, 1]),
                    "label": int(labels[index]),
                }
                for index in indices
            ],
            "components": [
                {
                    "index": index,
                    "weight": float(weights[index]) if index < weights.size else None,
                    **ellipse,
                }
                for index, ellipse in enumerate(ellipses)
            ],
            "ellipse_scale": 1.0,
        }
