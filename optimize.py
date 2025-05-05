from typing import Callable

from jaxtyping import Float
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import torch
from torch import Tensor

import util
from functions import ObjectiveSense


def optimize(
    fun: Callable[[Tensor], Float[Tensor, "n 1"]],
    bounds: Float[Tensor, "2 d"],
    num_initial_samples: int,
    sense: ObjectiveSense,
) -> OptimizeResult:
    r"""Optimize the function using L-BFGS-B algorithm.

    Args:
        fun: The objective function.
        bounds: Bounds for the optimization variables.
        num_initial_samples: Number of initial samples to try.
        sense: Whether to minimize or maximize the objective function.
    """

    def objective(x: np.ndarray) -> float:
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(1, -1)
        res = fun(x_tensor)
        return res.item() * sense.value

    def gradient(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(
            x.reshape(1, -1), dtype=torch.float32, requires_grad=True
        )
        res = fun(x_tensor)
        res.sum().backward()
        grad = x_tensor.grad
        assert grad is not None
        return grad.cpu().numpy().flatten() * sense.value

    # TODO: Multi starts

    candidates = util.uniform_sampling(num_initial_samples, bounds=bounds)
    cand_vals = fun(candidates) * sense.value
    best_idx = torch.argmin(cand_vals)
    x0 = candidates[best_idx].cpu().numpy()

    result: OptimizeResult = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds.transpose(0, 1).cpu().numpy(),
        options={"disp": True, "maxiter": 1000},
    )

    # Adjust the function value back for reporting
    result.fun = result.fun * sense.value

    return result
