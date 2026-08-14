"""Distributional metrics for misspecified Gaussian-mixture experiments."""

import numpy as np
from scipy.spatial.distance import cdist, pdist

from experiments.metrics.base_metric import BaseMetric
from experiments.metrics.metric_utils import log_gaussian_pdf, log_gm_pdf
from experiments.synthetic_data.gaussian import sample_gm_data


def _as_evaluation_sample(evaluation_X):
    evaluation_X = np.asarray(evaluation_X, dtype=float)
    if evaluation_X.ndim != 2 or evaluation_X.shape[0] < 2:
        raise ValueError(
            "evaluation_X must be a two-dimensional array with at least two rows"
        )
    if not np.all(np.isfinite(evaluation_X)):
        raise ValueError("evaluation_X must contain only finite values")
    return evaluation_X


def _sample_estimated_gmm(means, weights, covariances, num_samples, seed):
    samples, _ = sample_gm_data(
        means,
        weights,
        covariances,
        int(num_samples),
        seed=seed,
    )
    return samples


class HeldOutNegativeLogLikelihood(BaseMetric):
    """Cross-entropy of the fitted GMM on independent true samples."""

    def __init__(self, per_dimension=False):
        super().__init__("Held-out NLL")
        self.per_dimension = bool(per_dimension)

    def __call__(
        self,
        means,
        weights,
        covariances,
        evaluation_X,
        **kwargs,
    ):
        evaluation_X = _as_evaluation_sample(evaluation_X)
        value = -np.mean(log_gm_pdf(evaluation_X, means, weights, covariances))
        if self.per_dimension:
            value /= evaluation_X.shape[1]
        return float(value)


class TrueForwardKL(BaseMetric):
    """Monte Carlo forward KL using the exact true log-density."""

    def __init__(self, per_dimension=False):
        super().__init__("True Forward KL")
        self.per_dimension = bool(per_dimension)

    def __call__(
        self,
        means,
        weights,
        covariances,
        evaluation_X,
        evaluation_true_log_pdf,
        **kwargs,
    ):
        evaluation_X = _as_evaluation_sample(evaluation_X)
        true_log_pdf = np.asarray(evaluation_true_log_pdf, dtype=float).reshape(-1)
        if true_log_pdf.shape != (evaluation_X.shape[0],):
            raise ValueError(
                "evaluation_true_log_pdf must have one value per evaluation row"
            )
        if not np.all(np.isfinite(true_log_pdf)):
            raise ValueError("evaluation_true_log_pdf must contain finite values")

        estimated_log_pdf = log_gm_pdf(
            evaluation_X,
            means,
            weights,
            covariances,
        )
        value = np.mean(true_log_pdf - estimated_log_pdf)
        if self.per_dimension:
            value /= evaluation_X.shape[1]
        return float(value)


class TrueForwardKLStandardError(BaseMetric):
    """Standard error of the iid Monte Carlo forward-KL estimate."""

    def __init__(self, per_dimension=False):
        super().__init__("True KL MC SE")
        self.per_dimension = bool(per_dimension)

    def __call__(
        self,
        means,
        weights,
        covariances,
        evaluation_X,
        evaluation_true_log_pdf,
        **kwargs,
    ):
        evaluation_X = _as_evaluation_sample(evaluation_X)
        true_log_pdf = np.asarray(evaluation_true_log_pdf, dtype=float).reshape(-1)
        if true_log_pdf.shape != (evaluation_X.shape[0],):
            raise ValueError(
                "evaluation_true_log_pdf must have one value per evaluation row"
            )
        log_ratios = true_log_pdf - log_gm_pdf(
            evaluation_X,
            means,
            weights,
            covariances,
        )
        value = np.std(log_ratios, ddof=1) / np.sqrt(log_ratios.size)
        if self.per_dimension:
            value /= evaluation_X.shape[1]
        return float(value)


class ComponentConditionalNLL(BaseMetric):
    """Held-out component NLL after the evaluator's mean-based matching.

    Mixture weights are intentionally omitted: this score asks how well the
    Gaussian assigned to a true component describes observations from that
    component. Weight recovery is measured separately by ``WeightsTV``.
    """

    def __init__(self, per_dimension=False):
        super().__init__("Conditional NLL")
        self.per_dimension = bool(per_dimension)

    def __call__(
        self,
        means,
        covariances,
        evaluation_X,
        evaluation_labels,
        **kwargs,
    ):
        evaluation_X = _as_evaluation_sample(evaluation_X)
        labels = np.asarray(evaluation_labels, dtype=int).reshape(-1)
        if labels.shape != (evaluation_X.shape[0],):
            raise ValueError("evaluation_labels must match evaluation_X")
        if np.any(labels < 0) or np.any(labels >= len(means)):
            raise ValueError("evaluation_labels must be valid component indices")

        log_pdf = np.empty(evaluation_X.shape[0], dtype=float)
        for component in range(len(means)):
            mask = labels == component
            if np.any(mask):
                log_pdf[mask] = log_gaussian_pdf(
                    evaluation_X[mask],
                    means[component],
                    covariances[component],
                )
        value = -np.mean(log_pdf)
        if self.per_dimension:
            value /= evaluation_X.shape[1]
        return float(value)


class SlicedWasserstein(BaseMetric):
    """Empirical sliced Wasserstein-2 distance to fitted-GMM samples."""

    def __init__(
        self,
        num_samples=2_000,
        num_random_projections=100,
        include_skewness_directions=True,
        seed=17,
    ):
        super().__init__("Sliced W2")
        self.num_samples = int(num_samples)
        self.num_random_projections = int(num_random_projections)
        self.include_skewness_directions = bool(include_skewness_directions)
        self.seed = int(seed)
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.num_random_projections < 1:
            raise ValueError("num_random_projections must be positive")

    def __call__(
        self,
        means,
        weights,
        covariances,
        evaluation_X,
        skewness_directions=None,
        **kwargs,
    ):
        evaluation_X = _as_evaluation_sample(evaluation_X)
        n_samples = min(self.num_samples, evaluation_X.shape[0])
        rng = np.random.default_rng(self.seed)
        if n_samples == evaluation_X.shape[0]:
            true_samples = evaluation_X
        else:
            indices = rng.choice(
                evaluation_X.shape[0],
                size=n_samples,
                replace=False,
            )
            true_samples = evaluation_X[indices]
        estimated_samples = _sample_estimated_gmm(
            means,
            weights,
            covariances,
            n_samples,
            self.seed,
        )

        n_features = evaluation_X.shape[1]
        directions = rng.normal(
            size=(self.num_random_projections, n_features)
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        if self.include_skewness_directions and skewness_directions is not None:
            skewness_directions = np.asarray(skewness_directions, dtype=float)
            if (
                skewness_directions.ndim != 2
                or skewness_directions.shape[1] != n_features
            ):
                raise ValueError(
                    "skewness_directions must have one column per feature"
                )
            norms = np.linalg.norm(skewness_directions, axis=1)
            valid = np.isfinite(norms) & (norms > 0)
            if np.any(valid):
                extra_directions = (
                    skewness_directions[valid] / norms[valid, None]
                )
                directions = np.vstack([directions, extra_directions])

        true_projected = np.sort(true_samples @ directions.T, axis=0)
        estimated_projected = np.sort(
            estimated_samples @ directions.T,
            axis=0,
        )
        squared_w2_per_projection = np.mean(
            (true_projected - estimated_projected) ** 2,
            axis=0,
        )
        return float(np.sqrt(np.mean(squared_w2_per_projection)))


class EnergyDistance(BaseMetric):
    """Energy distance between training data and fitted-GMM samples.

    The biased V-statistic is used deliberately: it is exactly the energy
    distance between the two empirical measures and remains non-negative. The
    square root follows the conventional metric-scale definition.
    """

    def __init__(self, max_samples=None, seed=23):
        super().__init__("Energy Distance")
        self.max_samples = (
            None if max_samples is None else int(max_samples)
        )
        self.seed = int(seed)
        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError("max_samples must be at least 2")

    def __call__(self, means, weights, covariances, X, **kwargs):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must have at least two rows")

        rng = np.random.default_rng(self.seed)
        if self.max_samples is not None and X.shape[0] > self.max_samples:
            indices = rng.choice(
                X.shape[0],
                size=self.max_samples,
                replace=False,
            )
            true_samples = X[indices]
        else:
            true_samples = X

        estimated_samples = _sample_estimated_gmm(
            means,
            weights,
            covariances,
            true_samples.shape[0],
            self.seed,
        )
        cross_mean = np.mean(cdist(true_samples, estimated_samples))
        true_within_mean = (
            2.0 * np.sum(pdist(true_samples)) / true_samples.shape[0] ** 2
        )
        estimated_within_mean = (
            2.0
            * np.sum(pdist(estimated_samples))
            / estimated_samples.shape[0] ** 2
        )
        squared_distance = (
            2.0 * cross_mean - true_within_mean - estimated_within_mean
        )
        return float(np.sqrt(max(squared_distance, 0.0)))
