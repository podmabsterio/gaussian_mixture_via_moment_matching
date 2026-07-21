import os
from functools import partial

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from sklearn.cluster import KMeans

from src_np.bandwidth_selection import select_s_values_by_average_kernel_count
from src_torch.gmm.gmm_dimension_free import LOSS_EUCLIDEAN
from src_torch.gmm.gmm_dimension_free_linear import (
    GaussianMomentModelDimensionFreeLinear,
)
from src_torch.gmm.optimization import FullBatchStrategy, StochasticGradientStrategy
from src_torch.gmm.test_functions_utils.centers_generators import DataCentersGenerator
from src_torch.gmm.trainer import Trainer


MIN_SIGMA = 1e-6


class OracleTorchInitializer:
    def __init__(
        self,
        n_components,
        means,
        weights,
        covariances,
        means_noise_coef=0.0,
        sigmas_noise_coef=0.0,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.means = np.asarray(means, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.means_noise_coef = float(means_noise_coef)
        self.sigmas_noise_coef = float(sigmas_noise_coef)
        self.random_state = random_state

    def get_initial_params(self, X):
        rng = np.random.default_rng(self.random_state)
        _, d = X.shape

        means_noise = rng.normal(size=(self.n_components, d))
        sigma_noise = rng.normal(size=self.n_components)

        init_m = self.means + self.means_noise_coef * means_noise
        variances = np.diagonal(self.covariances, axis1=1, axis2=2).mean(axis=1)
        init_sigma = np.sqrt(variances) + self.sigmas_noise_coef * sigma_noise
        init_sigma = np.maximum(init_sigma, MIN_SIGMA)

        init_mu = self.weights / np.sum(self.weights)

        return {
            "init_m": torch.as_tensor(init_m, dtype=X.dtype, device=X.device),
            "init_sigma": torch.as_tensor(
                init_sigma,
                dtype=X.dtype,
                device=X.device,
            ),
            "init_mu": torch.as_tensor(init_mu, dtype=X.dtype, device=X.device),
        }


class KMeansTorchInitializer:
    def __init__(
        self,
        n_components,
        n_init=10,
        max_iter=300,
        random_state=None,
    ):
        self.n_components = int(n_components)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = random_state

    def get_initial_params(self, X):
        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.n_components,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            algorithm="lloyd",
        )
        labels_np = kmeans.fit_predict(X_np)
        means_np = kmeans.cluster_centers_.astype(float)

        labels = torch.as_tensor(labels_np, dtype=torch.long, device=X.device)
        means = torch.as_tensor(means_np, dtype=X.dtype, device=X.device)

        counts = torch.bincount(labels, minlength=self.n_components).to(dtype=X.dtype)
        weights = counts.clamp_min(1e-12)
        weights = weights / weights.sum()

        n_features = X.shape[1]
        global_center = X.mean(dim=0)
        global_variance = (X - global_center).square().sum(dim=1).mean() / n_features
        global_sigma = global_variance.clamp_min(MIN_SIGMA * MIN_SIGMA).sqrt()

        sigmas = torch.empty(self.n_components, dtype=X.dtype, device=X.device)
        for component_id in range(self.n_components):
            mask = labels == component_id
            if torch.any(mask):
                diff = X[mask] - means[component_id]
                variance = diff.square().sum(dim=1).mean() / n_features
                sigmas[component_id] = variance.clamp_min(MIN_SIGMA * MIN_SIGMA).sqrt()
            else:
                sigmas[component_id] = global_sigma

        return {
            "init_m": means,
            "init_sigma": sigmas.clamp_min(MIN_SIGMA),
            "init_mu": weights,
        }


class ExactCountBandwidthGenerator:
    def __init__(self, target_neighbor_counts):
        self.target_neighbor_counts = target_neighbor_counts

    def generate(self, X, centers, init_s=None, n_clusters=None):
        if n_clusters is None:
            raise ValueError("n_clusters must be provided")

        X_np = X.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        s_values = select_s_values_by_average_kernel_count(
            X_np,
            n_clusters,
            self.target_neighbor_counts,
            centers_np,
        )
        s = torch.as_tensor(s_values, dtype=X.dtype, device=X.device)
        return s[:, None].expand(-1, centers.shape[0])


class TorchDimensionFreeLinearWrapper:
    def __init__(
        self,
        n_components,
        target_neighbor_counts,
        init_mode="oracle",
        means_noise_coef=0.0,
        sigmas_noise_coef=0.0,
        num_directions=2,
        include_base_kernel=True,
        loss_mode=LOSS_EUCLIDEAN,
        relaxed_parametrization=False,
        precompute_Z=True,
        optimizer="adam",
        lr=1e-2,
        max_steps=500,
        batch_size=None,
        Z_batch_size=16,
        kmeans_n_init=10,
        kmeans_max_iter=300,
        dtype="float64",
        device="cpu",
        random_state=None,
    ):
        if init_mode not in {"oracle", "kmeans"}:
            raise ValueError("init_mode must be 'oracle' or 'kmeans'")
        self.n_components = int(n_components)
        self.target_neighbor_counts = target_neighbor_counts
        self.init_mode = init_mode
        self.means_noise_coef = float(means_noise_coef)
        self.sigmas_noise_coef = float(sigmas_noise_coef)
        self.num_directions = int(num_directions)
        self.include_base_kernel = bool(include_base_kernel)
        self.loss_mode = loss_mode
        self.relaxed_parametrization = bool(relaxed_parametrization)
        self.precompute_Z = bool(precompute_Z)
        self.optimizer = optimizer
        self.lr = float(lr)
        self.max_steps = int(max_steps)
        self.batch_size = batch_size
        self.Z_batch_size = Z_batch_size
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.dtype = dtype
        self.device = device
        self.random_state = random_state

        self.model = None
        self.trainer = None
        self.means_ = None
        self.weights_ = None
        self.covariances_ = None
        self.history_per_s = []

    def _torch_dtype(self):
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float64":
            return torch.float64
        raise ValueError("dtype must be 'float32' or 'float64'")

    def _optimizer_factory(self):
        if self.optimizer == "adam":
            return partial(torch.optim.Adam, lr=self.lr)
        if self.optimizer == "lbfgs":
            return partial(
                torch.optim.LBFGS,
                lr=self.lr,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )
        raise ValueError("optimizer must be 'adam' or 'lbfgs'")

    def _strategy(self):
        if self.batch_size is None:
            return FullBatchStrategy(
                optimizer_factory=self._optimizer_factory(),
                n_steps=self.max_steps,
            )

        return StochasticGradientStrategy(
            optimizer_factory=self._optimizer_factory(),
            n_steps=self.max_steps,
            batch_size=self.batch_size,
        )

    def fit(self, true_means, true_weights, true_covariances, X, **kwargs):
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        X_tensor = torch.as_tensor(
            X,
            dtype=self._torch_dtype(),
            device=torch.device(self.device),
        )

        if self.init_mode == "oracle":
            initializer = OracleTorchInitializer(
                n_components=self.n_components,
                means=true_means,
                weights=true_weights,
                covariances=true_covariances,
                means_noise_coef=self.means_noise_coef,
                sigmas_noise_coef=self.sigmas_noise_coef,
                random_state=self.random_state,
            )
        else:
            initializer = KMeansTorchInitializer(
                n_components=self.n_components,
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
                random_state=self.random_state,
            )

        self.model = GaussianMomentModelDimensionFreeLinear(
            centers_generator=DataCentersGenerator(),
            bandwidth_generator=ExactCountBandwidthGenerator(
                self.target_neighbor_counts,
            ),
            initializer=initializer,
            n_components=self.n_components,
            precompute_Z=self.precompute_Z,
            loss_mode=self.loss_mode,
            relaxed_parametrization=self.relaxed_parametrization,
            num_directions=self.num_directions,
            include_base_kernel=self.include_base_kernel,
        )
        self.trainer = Trainer(
            self.model,
            Z_batch_size=self.Z_batch_size,
            save_history=True,
            strategy=self._strategy(),
        )
        self.trainer.fit(X_tensor)

        self.means_ = self.trainer.means_.detach().cpu().numpy()
        self.weights_ = self.trainer.weights_.detach().cpu().numpy()
        covariances = self.trainer.covariances_.detach().cpu().numpy()
        if covariances.ndim == 1:
            d = self.means_.shape[1]
            covariances = covariances[:, None, None] * np.eye(d)
        self.covariances_ = covariances
        with torch.no_grad():
            final_loss = float(self.model.loss().detach().cpu().item())
        loss_history = list(self.trainer.loss_history)
        loss_history.append(final_loss)
        self.history_per_s = [np.asarray(loss_history, dtype=float)]
        return self

    def params_dict(self):
        return {
            "means": self.means_,
            "weights": self.weights_,
            "covariances": self.covariances_,
        }
