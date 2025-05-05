r"""
Nonlinear transform of negative of the Ackley function.

Reference:
    Appendix D.5. of Bayesian Optimization of Function Networks with Partial Evaluations
    https://proceedings.mlr.press/v235/buathong24a.html

"""

import torch
from jaxtyping import Float
from torch import Tensor
import networkx as nx
import numpy as np

from functions import Function, DAGFunction, Problem, ObjectiveSense


def original_ackley(a: float, b: float, c: float, x: Float[Tensor, "n d"]) -> Tensor:
    s1 = (x**2).mean(dim=-1, keepdim=True).sqrt()
    s2 = (c * x).cos().mean(dim=-1, keepdim=True)
    return -a * (-b * s1).exp() - s2.exp() + a + np.exp(1.0)


def ackley_f2(x: Tensor) -> Tensor:
    return -x * (5 * x / (6 * np.pi)).sin()


def ackley_f1(x: Tensor) -> Tensor:
    r"""Negative of the Ackley function."""
    return -original_ackley(20, 0.2, 2 * np.pi, x)


def get_ackley(ndim: int) -> Problem:
    name2func = {
        "f1": Function(
            func=ackley_f1, is_known=False, in_ndim=ndim, out_ndim=1, cost=1.0
        ),
        "f2": Function(func=ackley_f2, is_known=False, in_ndim=1, out_ndim=1, cost=1.0),
    }

    dag = nx.DiGraph()
    dag.add_node("f1", func="f1", raw_input=[(i, i) for i in range(ndim)])
    dag.add_node("f2", func="f2")
    dag.add_edge("f1", "f2", index=[(0, 0)])

    bounds = torch.tensor(
        [[-2.0 for _ in range(ndim)], [2.0 for _ in range(ndim)]], dtype=torch.double
    )

    return Problem(
        obj=DAGFunction(name2func=name2func, dag=dag),
        sense=ObjectiveSense.MAXIMIZE,
        bounds=bounds,
    )
