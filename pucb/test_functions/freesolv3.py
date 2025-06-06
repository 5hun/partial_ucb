r"""
References:
    Appendix D.3. of Bayesian Optimization of Function Networks with Partial Evaluations
    https://proceedings.mlr.press/v235/buathong24a.html

"""

from pathlib import Path

import torch
import networkx as nx
import polars as pl

from .. import gp
from ..functions import Function, FunctionNetwork, Problem, ObjectiveSense


def get_freesolv3(cost1: int, cost2: int) -> Problem:
    df = pl.read_csv(
        Path(__file__).parent / "freesolv_NN_rep3dim.csv", comment_prefix="#"
    )
    train_X1 = torch.tensor(
        df.select(pl.col("x1", "x2", "x3")).to_numpy(), dtype=torch.double
    )
    train_Y1 = torch.tensor(df.select(pl.col("cal")).to_numpy(), dtype=torch.double)
    train_Y2 = torch.tensor(df.select(pl.col("expt")).to_numpy(), dtype=torch.double)

    mod1 = gp.fit_gp_model(train_x=train_X1, train_y=train_Y1)
    mod2 = gp.fit_gp_model(train_x=train_Y1, train_y=train_Y2)
    name2func = {
        "f1": Function(
            func=gp.ExpectationFunction(mod1),
            is_known=False,
            in_ndim=3,
            out_ndim=1,
            cost=cost1,
        ),
        "f2": Function(
            func=gp.ExpectationFunction(mod2),
            is_known=False,
            in_ndim=1,
            out_ndim=1,
            cost=cost2,
        ),
    }

    dag = nx.DiGraph()
    dag.add_node("f1", func="f1", raw_input=[(i, i) for i in range(3)])
    dag.add_node("f2", func="f2")
    dag.add_edge("f1", "f2", index=[(0, 0)])

    bounds = torch.tensor(
        [[0.0 for _ in range(3)], [1.0 for _ in range(3)]], dtype=torch.double
    )

    return Problem(
        obj=FunctionNetwork(name2func=name2func, dag=dag),
        sense=ObjectiveSense.MAXIMIZE,
        bounds=bounds,
    )
