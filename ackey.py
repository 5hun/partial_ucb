import torch
from torch import Tensor
import networkx as nx
import numpy as np

from functions import Function, DAGFunction


def ackey_f2(x: Tensor) -> Tensor:
    return -x * (5 * x / (6 * np.pi)).sin()


def ackey_f1(x: Tensor) -> Tensor:
    return (
        20 * (-0.2 * x.norm(p=2, dim=-1, keepdim=True)).exp()
        + ((2 * np.pi * x).mean(dim=-1, keepdim=True).cos()).exp()
        - 20
        - np.exp(1.0)
    )


def get_ackey(ndim: int) -> DAGFunction:
    name2func = {
        "f1": Function(func=ackey_f1, is_known=False, in_ndim=ndim, out_ndim=1),
        "f2": Function(func=ackey_f2, is_known=False, in_ndim=1, out_ndim=1),
    }

    dag = nx.DiGraph()
    dag.add_node("f1", func="f1", raw_input=[(i, i) for i in range(ndim)])
    dag.add_node("f2", func="f2")
    dag.add_edge("f1", "f2", index=[(0, 0)])

    bounds = torch.tensor(
        [[-2.0 for _ in range(ndim)], [2.0 for _ in range(ndim)]], dtype=torch.double
    )

    return DAGFunction(name2func=name2func, dag=dag, bounds=bounds)
