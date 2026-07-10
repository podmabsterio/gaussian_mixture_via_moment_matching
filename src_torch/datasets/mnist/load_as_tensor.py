from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor
from torchvision import datasets, transforms
import torch

import matplotlib.pyplot as plt


def get_mnist_subset(
    digits: Optional[Sequence[int]] = None,
    max_size: Optional[int] = None,
    train: bool = True,
    root: str | Path = "./data",
    flatten: bool = True,
    normalize_to_unit_interval: bool = True,
    device: Optional[torch.device | str] = None,
) -> Tensor:
    if digits is not None:
        digits = list(digits)
        if len(digits) == 0:
            raise ValueError("digits не должен быть пустым.")
        invalid = [d for d in digits if d < 0 or d > 9]
        if invalid:
            raise ValueError(
                f"digits должен содержать числа от 0 до 9, получено: {invalid}"
            )

    if max_size is not None and max_size <= 0:
        raise ValueError(f"max_size должен быть > 0 или None, получено: {max_size}")

    root = Path(root)

    # Загружаем dataset без ToTensor, чтобы самим контролировать тип и форму.
    dataset = datasets.MNIST(
        root=str(root),
        train=train,
        download=True,
    )

    # dataset.data: [N, 28, 28], uint8
    # dataset.targets: [N], int64
    images: Tensor = dataset.data
    targets: Tensor = dataset.targets

    if digits is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        allowed = torch.tensor(digits, dtype=targets.dtype)
        mask = torch.isin(targets, allowed)

    subset = images[mask]

    if max_size is not None:
        subset = subset[:max_size]

    if normalize_to_unit_interval:
        subset = subset.float() / 255.0
    else:
        subset = subset.float()

    if flatten:
        subset = subset.view(subset.shape[0], 28 * 28)

    if device is not None:
        subset = subset.to(device)

    return subset


def show_mnist_vector(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    x = x.detach().cpu()

    if x.ndim == 2:
        if x.shape[0] != 1 or x.shape[1] != 28 * 28:
            raise ValueError(
                f"Ожидался тензор формы [1, 784], получено {tuple(x.shape)}"
            )
        x = x[0]
    elif x.ndim != 1:
        raise ValueError(
            f"Ожидался тензор формы [784] или [1, 784], получено {tuple(x.shape)}"
        )

    if x.numel() != 28 * 28:
        raise ValueError(f"Ожидался вектор длины 784, получено {x.numel()}")

    img = x.view(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
