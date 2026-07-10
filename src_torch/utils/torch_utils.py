import torch


def to_tensor(x, *, device=None, dtype=None):
    if x is None:
        return None
    return torch.as_tensor(x, device=device, dtype=dtype)
