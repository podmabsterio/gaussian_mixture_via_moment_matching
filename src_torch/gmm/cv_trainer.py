from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from sklearn.model_selection import KFold

from src.gmm.gmm import GaussianMomentModel
from src.gmm.optimization import OptimizationStrategy
from src.gmm.test_functions_utils.bandwidth_generators import (
    BaseBandwidthGenerator,
    DimensionAwareBandwidthGenerator,
)
from src.gmm.trainer import Trainer
from src.metrics.common import isotropic_covariances, log_prob_gmm


@dataclass(frozen=True)
class BandwidthCandidate:
    gamma: float
    ratio: float
    s_alpha: float
    s_beta: float
    generator: BaseBandwidthGenerator


class CVTrainer(Trainer):
    def __init__(
        self,
        model: GaussianMomentModel,
        bandwidth_count: int = 4,
        gamma_grid: Iterable[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        ratio_grid: Iterable[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
        cv: int = 5,
        shuffle: bool = True,
        random_state: int | None = 0,
        fit_random_state: int | None = 0,
        dimension_power: float = 0.5,
        effective_dimension: int | float | None = None,
        Z_batch_size=None,
        show_solution_path=False,
        mu_lb=None,
        save_history=False,
        strategy: OptimizationStrategy | None = None,
        eps: float = 1e-9,
    ):
        super().__init__(
            model=model,
            Z_batch_size=Z_batch_size,
            show_solution_path=show_solution_path,
            mu_lb=mu_lb,
            save_history=save_history,
            strategy=strategy,
        )
        if bandwidth_count < 1:
            raise ValueError("bandwidth_count must be positive")
        if cv < 2:
            raise ValueError("cv must be at least 2")

        self.bandwidth_count = bandwidth_count
        self.gamma_grid = tuple(float(value) for value in gamma_grid)
        self.ratio_grid = tuple(float(value) for value in ratio_grid)
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.fit_random_state = fit_random_state
        self.dimension_power = dimension_power
        self.effective_dimension = effective_dimension
        self.eps = eps

        self.cv_results_: list[dict] = []
        self.best_score_: float | None = None
        self.best_bandwidth_params_: dict | None = None
        self.best_bandwidth_generator_: BaseBandwidthGenerator | None = None

    def fit(self, X, y=None):
        X = torch.as_tensor(X)
        candidates = self._build_candidates()
        splitter = KFold(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )

        self.cv_results_ = []
        best_result = None

        for candidate_index, candidate in enumerate(candidates):
            fold_scores = []
            for fold_index, (train_idx, valid_idx) in enumerate(
                splitter.split(range(X.shape[0]))
            ):
                train_data = X[torch.as_tensor(train_idx, device=X.device)]
                valid_data = X[torch.as_tensor(valid_idx, device=X.device)]

                fold_model = self._clone_model_with_bandwidth(candidate.generator)
                fold_trainer = Trainer(
                    model=fold_model,
                    Z_batch_size=self.Z_batch_size,
                    show_solution_path=self.show_solution_path,
                    mu_lb=self.mu_lb,
                    save_history=False,
                    strategy=self.strategy,
                )
                with self._fold_rng_context(fold_index):
                    fold_trainer.fit(train_data)
                score = self._validation_log_likelihood(fold_trainer, valid_data)
                fold_scores.append(score)

            mean_score = sum(fold_scores) / len(fold_scores)
            result = {
                "candidate_index": candidate_index,
                "gamma": candidate.gamma,
                "ratio": candidate.ratio,
                "s_alpha": candidate.s_alpha,
                "s_beta": candidate.s_beta,
                "fold_scores": fold_scores,
                "mean_score": mean_score,
            }
            self.cv_results_.append(result)

            if best_result is None or mean_score > best_result["mean_score"]:
                best_result = result
                self.best_bandwidth_generator_ = candidate.generator

        self.best_score_ = best_result["mean_score"]
        self.best_bandwidth_params_ = {
            "gamma": best_result["gamma"],
            "ratio": best_result["ratio"],
            "s_alpha": best_result["s_alpha"],
            "s_beta": best_result["s_beta"],
            "bandwidth_count": self.bandwidth_count,
            "dimension_power": self.dimension_power,
            "effective_dimension": self.effective_dimension,
        }

        self.model.bandwidth_generator = self.best_bandwidth_generator_
        with self._fold_rng_context(self.cv):
            return super().fit(X, y=y)

    def _build_candidates(self) -> list[BandwidthCandidate]:
        if len(self.gamma_grid) == 0:
            raise ValueError("gamma_grid must contain at least one value")
        if len(self.ratio_grid) == 0:
            raise ValueError("ratio_grid must contain at least one value")

        ratios = (1.0,) if self.bandwidth_count == 1 else self.ratio_grid
        candidates = []

        for gamma in self.gamma_grid:
            if gamma <= 0:
                raise ValueError("All gamma_grid values must be positive")
            for ratio in ratios:
                if ratio < 1:
                    raise ValueError("All ratio_grid values must be >= 1")

                ratio_sqrt = ratio**0.5
                s_beta = gamma / ratio_sqrt
                s_alpha = gamma * ratio_sqrt
                generator = DimensionAwareBandwidthGenerator(
                    bandwidth_count=self.bandwidth_count,
                    s_alpha=s_alpha,
                    s_beta=s_beta,
                    dimension_power=self.dimension_power,
                    effective_dimension=self.effective_dimension,
                )
                candidates.append(
                    BandwidthCandidate(
                        gamma=gamma,
                        ratio=ratio,
                        s_alpha=s_alpha,
                        s_beta=s_beta,
                        generator=generator,
                    )
                )

        return candidates

    def _clone_model_with_bandwidth(self, bandwidth_generator):
        model = copy.deepcopy(self.model)
        model.bandwidth_generator = copy.deepcopy(bandwidth_generator)
        return model

    def _fold_rng_context(self, fold_index: int):
        return _fold_rng_context(self.fit_random_state, fold_index)

    def _validation_log_likelihood(
        self, trainer: Trainer, X_valid: torch.Tensor
    ) -> float:
        means = trainer.means_
        covariances = self._as_covariance_matrices(trainer.covariances_, means.shape[1])
        weights = trainer.weights_

        with torch.no_grad():
            log_likelihood = log_prob_gmm(
                X_valid,
                means,
                covariances,
                weights,
                eps=self.eps,
            ).mean()
        return float(log_likelihood.item())

    def _as_covariance_matrices(
        self, covariances: torch.Tensor, n_features: int
    ) -> torch.Tensor:
        if covariances.ndim == 3:
            return covariances
        if covariances.ndim == 1:
            sigmas = torch.sqrt(covariances.clamp_min(self.eps))
            return isotropic_covariances(sigmas, n_features)
        raise ValueError(
            "Fitted covariances must have shape [K] for isotropic variances "
            f"or [K, D, D], got {tuple(covariances.shape)}"
        )


@contextmanager
def _fold_rng_context(fit_random_state: int | None, fold_index: int):
    if fit_random_state is None:
        yield
        return

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(fit_random_state) + int(fold_index))
        yield
