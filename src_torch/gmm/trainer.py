from src_torch.gmm.gmm import GaussianMomentModel
from src_torch.gmm.optimization import OptimizationHistory, OptimizationStrategy


EPS = 1e-8


class Trainer:
    def __init__(
        self,
        model: GaussianMomentModel,
        Z_batch_size=None,
        show_solution_path=False,
        mu_lb=None,
        save_history=False,
        strategy: OptimizationStrategy | None = None,
    ):
        if strategy is None:
            raise ValueError("Trainer requires an optimization strategy object.")

        self.model = model
        self.Z_batch_size = Z_batch_size
        self.show_solution_path = show_solution_path
        self.save_history = save_history
        self.mu_lb = mu_lb
        self.strategy = strategy
        self.history = OptimizationHistory(enabled=save_history)
        self.grad_history = self.history.grad_history
        self.loss_history = self.history.loss_history
        self.block_history = self.history.block_history

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None

    def _extract_fitted_parameters(self):
        post_fit = getattr(self.model, "post_fit", None)
        if post_fit is not None:
            post_fit()

        self.means_ = self.model.m.detach()
        if hasattr(self.model, "fitted_covariances"):
            self.covariances_ = self.model.fitted_covariances().detach()
        elif hasattr(self.model, "fitted_isotropic_variances"):
            self.covariances_ = self.model.fitted_isotropic_variances().detach()
        else:
            self.covariances_ = self.model.sigma.detach() ** 2
        self.weights_ = self.model.mu.detach()

        if self.mu_lb is not None:
            self.weights_[self.weights_ < self.mu_lb] = 0
            self.weights_ /= self.weights_.sum().clamp_min(1e-12)

    def fit(self, X, y=None):
        self.model.insert_data(X, batch_size=self.Z_batch_size)
        self.strategy.fit(self.model, self.history)
        self._extract_fitted_parameters()

        return self
