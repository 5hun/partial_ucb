r"""
Gaussian Process (GP) model for Bayesian optimization.
"""

from jaxtyping import Float
import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def ucb(mod: SingleTaskGP, x: Tensor, alpha: Tensor) -> Tensor:
    r"""
    Compute the Upper Confidence Bound (UCB) for a given model and input.
    Args:
        mod: SingleTaskGP
            The Gaussian process model.
        x: Tensor
            The input data.
        alpha: Tensor
            The exploration parameter.
    Returns:
        Tensor
            The UCB value.
    """

    posterior = mod.posterior(x)
    mean = posterior.mean
    std = posterior.variance.sqrt()
    return mean + alpha * std


def get_model(
    train_x: Float[Tensor, "n d"], train_y: Float[Tensor, "n 1"], train_yvar: float
) -> SingleTaskGP:
    r"""
    Fit a Gaussian process model to the data.
    Args:
        train_x: Training data.
        train_y: Training targets.
        train_yvar: Variance of the training targets.
    Returns:
        model: Fitted Gaussian process model.

    """

    model = SingleTaskGP(
        train_x,
        train_y,
        train_Yvar=train_yvar * torch.ones_like(train_y),
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


class ExpectationFunction:
    def __init__(self, mod: Model):
        self.mod = mod

    def __call__(self, x: Tensor) -> Tensor:
        return self.mod.posterior(x).mean
