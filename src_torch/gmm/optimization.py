from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def get_grad_norm(model, norm_type=2.0):
    grads = [p.grad.detach() for p in trainable_parameters(model) if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    return torch.nn.utils.get_total_norm(grads, norm_type=norm_type).item()


def snapshot_parameters(params):
    return [param.detach().clone() for param in params]


def restore_parameters(params, snapshot):
    with torch.no_grad():
        for param, value in zip(params, snapshot):
            param.copy_(value)


def parameters_are_finite(params):
    return all(bool(torch.isfinite(param).all().item()) for param in params)


def tensor_is_finite(value):
    if not torch.is_tensor(value):
        return True
    return bool(torch.isfinite(value.detach()).all().item())


@torch.no_grad()
def model_loss_is_finite(model):
    try:
        loss = model.loss()
    except (
        Exception
    ):  # noqa: BLE001 - non-finite guards should not mask the original optimizer API.
        return False
    return tensor_is_finite(loss)


class IndexDataset(Dataset):
    def __init__(self, n_samples):
        self.indices = torch.arange(n_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


class OptimizationHistory:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.loss_history = []
        self.grad_history = []
        self.block_history = []

    def record(self, model, loss, block=None):
        if not self.enabled:
            return

        self.grad_history.append(get_grad_norm(model))
        self.loss_history.append(float(loss.detach().item()))
        self.block_history.append(block)


class OptimizationStrategy:
    def fit(self, model, history: OptimizationHistory):
        raise NotImplementedError


class FullBatchStrategy(OptimizationStrategy):
    def __init__(self, optimizer_factory=None, n_steps=1000, verbose=False):
        self.optimizer_factory = optimizer_factory
        self.n_steps = n_steps
        self.verbose = verbose

    def fit(self, model, history: OptimizationHistory):
        optimizer_factory = self.optimizer_factory or optim.LBFGS
        params = trainable_parameters(model)
        optimizer = optimizer_factory(params)
        log_every = max(1, self.n_steps // 10)

        for step in range(self.n_steps):
            snapshot = snapshot_parameters(params)

            def closure():
                optimizer.zero_grad()
                loss = model.loss()
                loss.backward()
                return loss

            loss = optimizer.step(closure)

            if (
                not tensor_is_finite(loss)
                or not parameters_are_finite(params)
                or not model_loss_is_finite(model)
            ):
                restore_parameters(params, snapshot)
                restored_loss = model.loss()
                history.record(model, restored_loss)
                if self.verbose:
                    print(
                        "Stopping optimization: restored last finite parameters "
                        f"after non-finite step {step}"
                    )
                break

            history.record(model, loss)

            if self.verbose and step % log_every == 0:
                print(f"step {step} loss {loss.item():.6f}")


class StochasticGradientStrategy(OptimizationStrategy):
    def __init__(
        self,
        optimizer_factory=None,
        n_steps=1000,
        batch_size=32,
        scheduler_factory=None,
        verbose=False,
    ):
        self.optimizer_factory = optimizer_factory
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.scheduler_factory = scheduler_factory
        self.verbose = verbose

    def fit(self, model, history: OptimizationHistory):
        optimizer_factory = self.optimizer_factory or optim.Adam
        optimizer = optimizer_factory(trainable_parameters(model))
        scheduler = (
            self.scheduler_factory(optimizer)
            if self.scheduler_factory is not None
            else None
        )
        loader = make_index_loader(model, self.batch_size)
        log_every = max(1, self.n_steps // 10)

        for step in range(self.n_steps):
            loss = None
            for index_batch in loader:
                optimizer.zero_grad()
                loss = model.loss(index_batch)
                loss.backward()
                history.record(model, loss)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if self.verbose and step % log_every == 0 and loss is not None:
                print(f"step {step} loss {loss.item():.6f}")


class BlockCoordinateDescentStrategy(OptimizationStrategy):
    def __init__(
        self,
        blocks=None,
        outer_steps=100,
        batch_size=None,
        verbose=False,
    ):
        self.blocks = blocks or default_bcd_blocks()
        self.outer_steps = outer_steps
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, model, history: OptimizationHistory):
        all_params = trainable_parameters(model)
        log_every = max(1, self.outer_steps // 10)

        for outer_step in range(self.outer_steps):
            loss = None
            for block_cfg in self.blocks:
                block_name = block_cfg.get("name")
                params = resolve_block_params(model, block_name)
                steps = int(block_cfg.get("steps", 1))

                with_only_trainable(all_params, params)
                try:
                    loss = optimize_param_block(
                        model=model,
                        params=params,
                        optimizer_factory=block_cfg.get("optimizer_factory"),
                        steps=steps,
                        batch_size=block_cfg.get("batch_size", self.batch_size),
                        history=history,
                        block_name=block_name,
                    )
                finally:
                    set_requires_grad(all_params, True)

            if self.verbose and outer_step % log_every == 0 and loss is not None:
                print(f"outer_step {outer_step} loss {loss.item():.6f}")


def make_index_loader(model, batch_size):
    num_index_functions = model.num_index_functions()
    index_dataset = IndexDataset(num_index_functions)
    return DataLoader(index_dataset, batch_size=batch_size, shuffle=True)


def trainable_parameters(model):
    return unique_params(model.parameter_blocks()["all"])


def optimize_param_block(
    model,
    params,
    optimizer_factory,
    steps,
    batch_size,
    history,
    block_name,
):
    optimizer_factory = optimizer_factory or optim.Adam
    optimizer = optimizer_factory(params)

    loss = None
    for _ in range(steps):
        if isinstance(optimizer, optim.LBFGS):

            def closure():
                optimizer.zero_grad()
                block_loss = model.loss()
                block_loss.backward()
                return block_loss

            loss = optimizer.step(closure)
            history.record(model, loss, block=block_name)
        elif batch_size is None:
            optimizer.zero_grad()
            loss = model.loss()
            loss.backward()
            history.record(model, loss, block=block_name)
            optimizer.step()
        else:
            for index_batch in make_index_loader(model, batch_size):
                optimizer.zero_grad()
                loss = model.loss(index_batch)
                loss.backward()
                history.record(model, loss, block=block_name)
                optimizer.step()

    return loss


def default_bcd_blocks():
    return [
        {"name": "weights", "steps": 1},
        {"name": "means", "steps": 1},
        {"name": "scales", "steps": 1},
    ]


def resolve_block_params(model, block_name):
    if block_name is None:
        raise ValueError("BCD block config must define a block name")

    blocks = model.parameter_blocks()
    names = block_name if isinstance(block_name, list) else [block_name]

    params = []
    for name in names:
        if name not in blocks:
            available = ", ".join(sorted(blocks))
            raise ValueError(
                f"Unknown parameter block '{name}'. Available blocks: {available}"
            )
        params.extend(blocks[name])

    return unique_params(params)


def unique_params(params):
    seen = set()
    unique = []
    for param in params:
        marker = id(param)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(param)
    return unique


def with_only_trainable(all_params, trainable_params):
    trainable_ids = {id(param) for param in trainable_params}
    for param in all_params:
        param.requires_grad_(id(param) in trainable_ids)


def set_requires_grad(params: Iterable[torch.nn.Parameter], requires_grad: bool):
    for param in params:
        param.requires_grad_(requires_grad)
