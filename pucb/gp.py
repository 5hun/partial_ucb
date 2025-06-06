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


def fit_gp_model(
    train_x: Float[Tensor, "n d"],
    train_y: Float[Tensor, "n 1"],
    train_yvar: float | None = None,
    state_dict: None | dict = None,
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
        train_x.detach(),
        train_y.detach(),
        train_Yvar=(
            train_yvar * torch.ones_like(train_y) if train_yvar is not None else None
        ),
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(train_x)

    if state_dict is not None:
        drop = ("input_transform", "outcome_transform")
        warm_sd = {k: v for k, v in state_dict.items() if not k.startswith(drop)}
        model.load_state_dict(warm_sd, strict=False)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    fit_gpytorch_mll(mll, max_attempts=10, pick_best_of_all_attempts=True)

    model.eval()
    return model


class ExpectationFunction:
    def __init__(self, mod: SingleTaskGP):
        self.mod = mod

    def __call__(self, x: Tensor) -> Tensor:
        return self.mod.posterior(x).mean


class UCBFunction:
    def __init__(self, mod: SingleTaskGP, alpha: float, lower: bool):
        self.mod = mod
        self.alpha = alpha
        self.lower = lower

    def __call__(self, x: Tensor) -> Tensor:
        posterior = self.mod.posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        if self.lower:
            return mean - self.alpha * std
        else:
            return mean + self.alpha * std
