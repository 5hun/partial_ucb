import torch
import networkx as nx

from ..problem import Function, FunctionNetwork, Problem, ObjectiveSense


def get_norm(ndim: int, p: int) -> Problem:
    functions = {
        "f1": Function(
            func=lambda x: x.norm(p=p, dim=-1, keepdim=True),
            is_known=False,
            in_ndim=ndim,
            out_ndim=1,
            cost=1,
        ),
    }
    dag = nx.DiGraph()
    dag.add_node("f1", func="f1", raw_input=[(i, i) for i in range(ndim)])
    bounds = torch.tensor(
        [[-1.0 for _ in range(ndim)], [1.0 for _ in range(ndim)]], dtype=torch.double
    )
    return Problem(
        obj=FunctionNetwork(functions=functions, dag=dag),
        sense=ObjectiveSense.MINIMIZE,
        bounds=bounds,
    )
