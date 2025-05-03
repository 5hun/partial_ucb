from typing import Callable

from jaxtyping import Float
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import torch
from torch import Tensor


def optimize(
    fun: Callable[[Tensor], Float[Tensor, "n 1"]], bounds: Float[Tensor, "2 d"]
) -> OptimizeResult:
    r"""Optimize the function using L-BFGS-B algorithm."""

    def objective(x: np.ndarray) -> float:
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        res = fun(x_tensor)
        return res.item()

    def gradient(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(
            x.reshape(1, -1), dtype=torch.float32, requires_grad=True
        )
        res = fun(x_tensor)
        res.sum().backward()
        grad = x_tensor.grad
        assert grad is not None
        return grad.cpu().numpy().flatten()

    np_bounds = bounds.cpu().numpy()

    # TODO: More sophisticated initialization, and multi starts.
    x0 = (np_bounds[0] + np_bounds[1]) / 2

    result: OptimizeResult = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds.transpose(0, 1).cpu().numpy(),
        options={"disp": True},
    )

    return result
