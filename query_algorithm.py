from dataclasses import dataclass

import torch
from torch import Tensor

import botorch
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction

from functions import DAGFunction, Function
import gp


@dataclass(frozen=True)
class QueryResponse:
    query_function: str
    query_input: Tensor
    info: dict


class QueryAlgorithm:
    def step(self, data: dict[str, tuple[Tensor, Tensor]]) -> QueryResponse:
        raise NotImplementedError()

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        raise NotImplementedError()


class DAGUCB:
    def __init__(
        self,
        fun: DAGFunction,
        alpha: float,
        mods: dict[str, botorch.models.SingleTaskGP],
    ):
        self.fun = fun
        self.alpha = alpha
        self.mods = mods

        self.noise_info = self._get_noise_info()
        num_noise = sum([y - x for x, y in self.noise_info.values()])
        self.bounds = torch.cat(
            [
                fun.bounds,
                torch.tensor([[-1] * num_noise, [1] * num_noise], dtype=torch.float32),
            ],
            dim=1,
        )

    def _get_noise_info(self) -> dict[str, tuple[int, int]]:
        index = 0
        noise_info = {}
        for v in self.fun.dag.nodes:
            funcname = self.fun.dag.nodes[v]["func"]
            func = self.fun.name2func[funcname]
            if not func.is_known:
                noise_info[v] = (index, index + func.out_ndim)
                index += func.out_ndim
        return noise_info

    def _eval_node(self, node: str, cache: dict, x: Tensor):
        for m in self.fun.dag.predecessors(node):
            if m not in cache:
                self._eval_node(m, cache, x)
            assert m in cache

        inputs_concat = self.fun.get_input_index(node)
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

        func = self.fun.dag.nodes[node]["func"]
        if func.is_known:
            res = self.fun.eval_sub(func, y)
        else:
            x_idx, x_idx_end = self.noise_info[node]
            noise = self.alpha * x[:, x_idx:x_idx_end]
            res = gp.ucb(self.mods[node], y, noise)

        cache[node] = res

    def eval(self, x: Tensor) -> dict[str, Tensor]:
        cache = {}
        res_node = self.fun.get_output_node_name()
        self._eval_node(res_node, cache, x)
        return cache


class DAGUCBAcquisitionFunction(AcquisitionFunction):
    def __init__(self, dagucb: DAGUCB):
        dummy_model = botorch.models.SingleTaskGP(
            torch.empty(0, 0),
            torch.empty(0, 0),
        )
        super().__init__(model=dummy_model)
        self.dagucb = dagucb

    def forward(self, X: Tensor) -> Tensor:
        cache = self.dagucb.eval(X)
        return cache[self.dagucb.fun.get_output_node_name()]


class DAGAcquisitionFunction(AcquisitionFunction):
    def __init__(self, dag: DAGFunction):
        dummy_model = botorch.models.SingleTaskGP(
            torch.empty(0, 0),
            torch.empty(0, 0),
        )
        super().__init__(model=dummy_model)
        self.dag = dag

    def forward(self, X: Tensor) -> Tensor:
        cache = self.dag.eval(X)
        return cache[self.dag.get_output_node_name()]


class PartialUCB(QueryAlgorithm):
    def __init__(self, fun: DAGFunction, alpha: float, train_yvar: float = 1e-4):
        self.fun = fun
        self.alpha = alpha
        self.train_yvar = train_yvar

    def _optimize_dag_function(
        self, models: dict[str, botorch.models.SingleTaskGP]
    ) -> Tensor:
        dagucb = DAGUCB(self.fun, self.alpha, models)
        bounds = dagucb.bounds
        acqf = DAGUCBAcquisitionFunction(dagucb)
        result, _acqf_val = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        return result

    def step(self, data: dict[str, tuple[Tensor, Tensor]]) -> QueryResponse:
        mods = {
            nm: gp.get_model(x, y, train_yvar=self.train_yvar)
            for nm, (x, y) in data.items()
        }
        result = self._optimize_dag_function(mods)

        name2func = self.fun.name2func.copy()
        for nm in mods:
            func = name2func[nm]
            name2func[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(mods[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = DAGFunction(name2func, self.fun.dag, self.fun.bounds)
        eval_cache = tmp_fun.eval(result)
        res_node = tmp_fun.get_output_node_name()
        eval_cache[res_node].backward()

        best_r = float("-inf")
        best_response = None
        for nm in self.fun.name2func:
            func = self.fun.name2func[nm]
            if func.is_known:
                continue
            tmp_res = eval_cache[nm]
            tmp_grad = tmp_res.grad
            assert tmp_grad is not None
            assert tmp_grad.shape == tmp_res.shape
            tmp_input = tmp_fun.get_input_tensor(nm, eval_cache, result)
            cov = mods[nm].forward(tmp_input).covariance_matrix
            print(f"{tmp_input.shape=}, {tmp_grad.shape=}, {cov.shape=}")
            r = tmp_grad @ (cov @ tmp_grad)
            if r > best_r:
                best_r = r
                best_response = QueryResponse(
                    query_function=nm, query_input=tmp_input, info={"r": r}
                )
        assert best_response is not None
        return best_response

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        mods = {
            nm: gp.get_model(x, y, train_yvar=self.train_yvar)
            for nm, (x, y) in data.items()
        }
        name2func = self.fun.name2func.copy()
        for nm in mods:
            func = name2func[nm]
            name2func[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(mods[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = DAGFunction(name2func, self.fun.dag, self.fun.bounds)
        acqf = DAGAcquisitionFunction(tmp_fun)
        bounds = tmp_fun.bounds
        result, acqf_val = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        return result, acqf_val.item()
