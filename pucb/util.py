import torch
from torch import Tensor


def uniform_sampling(n: int, bounds: Tensor) -> Tensor:
    return (bounds[1] - bounds[0]) * torch.rand(
        n, bounds[0].numel(), device=bounds.device
    ) + bounds[0]
