from dataclasses import dataclass
from typing import Callable

from jaxtyping import Float
import torch
from torch import Tensor
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class Function:
    is_known: bool
    func: Callable[[Tensor], Tensor]
    in_ndim: int
    out_ndim: int

    # TODO: Allow func can be None for unknown functions


class DAGFunction:
    def __init__(self, name2func: dict[str, Function], dag: nx.DiGraph, bounds: Tensor):
        self.name2func = name2func
        self.dag = dag
        self.bounds = bounds

    def eval_sub(self, name: str, x: Float[Tensor, "n d"]) -> Float[Tensor, "n d2"]:
        return self.name2func[name].func(x)

    def get_output_node_name(self) -> str:
        no_out_deg_node = [
            node for node in self.dag.nodes if self.dag.out_degree(node) == 0
        ]
        assert (
            len(no_out_deg_node) == 1
        ), "There should be only one node with no out degree"
        return no_out_deg_node[0]

    def get_input_index(self, node: str) -> list[tuple[str, int, int]]:
        inputs = []
        for m in self.dag.predecessors(node):
            for x_idx, y_idx in self.dag.edges[m, node]["index"]:
                inputs.append((m, x_idx, y_idx))

        if "raw_input" in self.dag.nodes[node]:
            for x_idx, y_idx in self.dag.nodes[node]["raw_input"]:
                inputs.append(("__raw_input__", x_idx, y_idx))

        assert len(inputs) > 0
        inputs.sort(key=lambda x: (x[2], x[1], x[0]))
        assert inputs[0][-1] == 0
        assert len(inputs) == inputs[-1][-1] + 1
        inputs = [(nm, x_idx) for nm, x_idx, y_idx in inputs]
        inputs_concat = []
        for nm, x_idx in inputs:
            if (
                len(inputs_concat) == 0
                or inputs_concat[-1][0] != nm
                or inputs_concat[-1][1] + 1 < x_idx
            ):
                inputs_concat.append((nm, x_idx, x_idx + 1))
            else:
                inputs_concat[-1] = (
                    inputs_concat[-1][0],
                    inputs_concat[-1][1],
                    x_idx + 1,
                )
        return inputs_concat

    def get_input_tensor(
        self, node: str, cache: dict[str, Tensor], x: Tensor
    ) -> Tensor:
        inputs_concat = self.get_input_index(node)
        y = []
        for nm, x_idx, x_idx_end in inputs_concat:
            if nm == "__raw_input__":
                y.append(x[:, x_idx:x_idx_end])
            else:
                if cache[nm].shape[-1] == 1:
                    assert x_idx == 0 and x_idx_end == 1
                    y.append(cache[nm])
                else:
                    y.append(cache[nm][:, x_idx:x_idx_end])
        y = torch.cat(y, dim=-1)
        return y

    def _eval_node(self, node: str, cache: dict[str, Tensor], x: Tensor) -> None:
        for m in self.dag.predecessors(node):
            if m not in cache:
                self._eval_node(m, cache, x)
            assert m in cache

        y = self.get_input_tensor(node, cache, x)

        func = self.dag.nodes[node]["func"]
        res = self.eval_sub(func, y)
        cache[node] = res

    def eval(self, x: Float[Tensor, "n d"]) -> dict[str, Tensor]:
        cache = {}
        res_node = self.get_output_node()
        self._eval_node(res_node, cache, x)
        return cache

    def get_output_node(self) -> str:
        r"""Get the output node of the DAG."""
        no_out_deg_node = [
            node for node in self.dag.nodes if self.dag.out_degree(node) == 0
        ]
        assert (
            len(no_out_deg_node) == 1
        ), "There should be only one node with no out degree"
        return no_out_deg_node[0]

    def __call__(self, x: Float[Tensor, "n_d"]) -> Float[Tensor, "n 1"]:
        return self.eval(x)[self.get_output_node()]
