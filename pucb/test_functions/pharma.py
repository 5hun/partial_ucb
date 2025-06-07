r"""
References:
    Appendix D.4. of Bayesian Optimization of Function Networks with Partial Evaluations
    https://proceedings.mlr.press/v235/buathong24a.html

"""

import torch
from jaxtyping import Float
from torch import Tensor
import networkx as nx

from ..functions import Function, FunctionNetwork, Problem, ObjectiveSense


def simple_nn(
    x: Float[Tensor, "n d"],
    coefs: Float[Tensor, "m d"],
    weights: Float[Tensor, "m"],
    bias: float,
) -> Float[Tensor, "n 1"]:
    assert x.shape[1] + 1 == coefs.shape[1]
    assert coefs.shape[0] == weights.shape[0]
    invexp = torch.sigmoid(coefs[:, 0] + torch.matmul(x, coefs[:, 1:].T))
    ret = bias + torch.mv(invexp, weights)
    assert ret.shape == (x.shape[0],)
    return ret.reshape(-1, 1)


F1_COEFS = torch.tensor(
    [
        [0.32, 5.06, -4.07, -0.36, -0.34],
        [-4.83, 7.43, 3.46, 9.19, 16.58],
        [7.90, 7.91, 4.48, 4.08, 8.28],
        [9.41, -7.99, 0.65, 3.14, 0.31],
    ],
    dtype=torch.double,
)


F1_WEIGHTS = torch.tensor([9.20, 9.88, 10.84, 15.18], dtype=torch.double)


def f1(x: Float[Tensor, "n 4"]) -> Float[Tensor, "n 1"]:
    return simple_nn(x=x, coefs=F1_COEFS, weights=F1_WEIGHTS, bias=-3.95)


F2_COEFS = torch.tensor(
    [
        [3.05, 0.03, -0.16, 4.03, -0.54],
        [1.78, 0.60, -3.19, 0.10, 0.54],
        [0.01, 2.04, -3.73, 0.10, -1.05],
        [1.82, 4.78, 0.48, -4.68, -1.65],
        [2.69, 5.99, 3.87, 3.10, -2.17],
    ],
    dtype=torch.double,
)


F2_WEIGHTS = torch.tensor([0.62, 0.65, -0.72, -0.45, -0.32], dtype=torch.double)


def f2(x: Float[Tensor, "n 4"]) -> Float[Tensor, "n 1"]:
    return simple_nn(x=x, coefs=F2_COEFS, weights=F2_WEIGHTS, bias=1.07)


def f3(x: Float[Tensor, "n 2"]) -> Float[Tensor, "n 1"]:
    x1 = x[:, 0]
    x2 = x[:, 1]
    ret = (60 - x1) / 60 * x2 / 1.5
    return ret.reshape(-1, 1)


def get_pharma() -> Problem:
    name2func = {
        "f1": Function(func=f1, is_known=False, in_ndim=4, out_ndim=1, cost=1),
        "f2": Function(func=f2, is_known=False, in_ndim=4, out_ndim=1, cost=49),
        "f3": Function(func=f3, is_known=True, in_ndim=2, out_ndim=1),
    }

    dag = nx.DiGraph()
    dag.add_node("f1", func="f1", raw_input=[(i, i) for i in range(4)])
    dag.add_node("f2", func="f2", raw_input=[(i, i) for i in range(4)])
    dag.add_node("f3", func="f3")
    dag.add_edge("f1", "f3", index=[(0, 0)])
    dag.add_edge("f2", "f3", index=[(0, 1)])

    bounds = torch.tensor(
        [[-1.0 for _ in range(4)], [1.0 for _ in range(4)]], dtype=torch.double
    )

    return Problem(
        obj=FunctionNetwork(name2func=name2func, dag=dag),
        sense=ObjectiveSense.MAXIMIZE,
        bounds=bounds,
    )
