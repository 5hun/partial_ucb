r"""
Gaussian Process (GP) model for Bayesian optimization.
"""

from torch import Tensor
from torch.models import Model, SingleTaskGP
from torch.models.transform.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


def ucb(mod: Model, x: Tensor, alpha: Tensor) -> Tensor:
    r"""
    Compute the Upper Confidence Bound (UCB) for a given model and input.
    Args:
        mod: Model
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


def get_model(train_x: Tensor, train_y: Tensor, train_yvar: float) -> Model:
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
        train_yvar=train_yvar.expand_as(train_y),
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model
