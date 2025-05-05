from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor

import botorch

from functions import DAGFunction, Function, Problem
import gp
import optimize


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
        problem: Problem,
        alpha: float,
        mods: dict[str, botorch.models.SingleTaskGP],
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.mods = mods

        self.noise_info = self._get_noise_info()
        num_noise = sum([y - x for x, y in self.noise_info.values()])
        self.bounds = torch.cat(
            [
                problem.bounds,
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

        func_name: str = self.fun.dag.nodes[node]["func"]
        func = self.fun.name2func[func_name]
        if func.is_known:
            res = self.fun.eval_sub(func_name, y)
        else:
            noise_offset = self.problem.bounds.shape[1]
            x_idx, x_idx_end = self.noise_info[node]
            x_idx += noise_offset
            x_idx_end += noise_offset
            noise = self.alpha * x[:, x_idx:x_idx_end]
            res = gp.ucb(self.mods[node], y, noise)

        cache[node] = res

    def eval(self, x: Tensor) -> dict[str, Tensor]:
        cache = {}
        res_node = self.fun.get_output_node_name()
        self._eval_node(res_node, cache, x)
        return cache

    def __call__(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n 1"]:
        res = self.eval(x)
        return res[self.fun.get_output_node_name()]


class PartialUCB(QueryAlgorithm):
    def __init__(self, problem: Problem, alpha: float, train_yvar: float = 1e-5):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.train_yvar = train_yvar
        self.previous_sols = None

    def _optimize_dagucb(
        self, models: dict[str, botorch.models.SingleTaskGP]
    ) -> tuple[Tensor, Tensor, float]:
        dagucb = DAGUCB(self.problem, self.alpha, models)
        res = optimize.optimize(
            fun=dagucb,
            bounds=dagucb.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        print(f"{res=}")
        assert res.success
        x_org = torch.tensor(res.x).reshape(1, -1)
        x = x_org[:, : self.problem.bounds.shape[1]]
        noise = x_org[:, self.problem.bounds.shape[1] :]
        fval = res.fun
        return x, noise, fval

    def step(self, data: dict[str, tuple[Tensor, Tensor]]) -> QueryResponse:
        mods = {
            nm: gp.get_model(x, y, train_yvar=self.train_yvar)
            for nm, (x, y) in data.items()
        }
        result, _noise, _result_fval = self._optimize_dagucb(mods)
        print(f"Optimize DAGUCB result: {result=}")

        name2func = self.fun.name2func.copy()
        for nm in mods:
            func = name2func[nm]
            name2func[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(mods[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = DAGFunction(name2func, self.fun.dag)

        eval_cache = tmp_fun.eval(result)
        res_node = tmp_fun.get_output_node_name()

        for nm in self.fun.name2func:
            func = self.fun.name2func[nm]
            if func.is_known:
                continue
            tmp_res = eval_cache[nm]
            tmp_res.retain_grad()
            assert tmp_res.requires_grad
            assert tmp_res.grad_fn is not None

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
            cov = mods[nm].posterior(tmp_input).covariance_matrix
            r = (tmp_grad @ (cov @ tmp_grad)).item() / func.cost
            assert r >= 0
            if r > best_r:
                best_r = r
                best_response = QueryResponse(
                    query_function=nm, query_input=tmp_input.detach(), info={"r": r}
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
        tmp_fun = DAGFunction(name2func, self.fun.dag)
        res = optimize.optimize(
            tmp_fun,
            self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        assert res.success
        sol = torch.tensor(res.x).reshape(1, -1)
        fval = res.fun

        if self.previous_sols is None:
            self.previous_sols = sol
        else:
            self.previous_sols = torch.cat((self.previous_sols, sol), dim=0)

        if self.previous_sols.shape[0] > 0:
            with torch.no_grad():
                prev_vals = tmp_fun(self.previous_sols) * self.problem.sense.value
                best_idx = torch.argmin(prev_vals)
                sol = self.previous_sols[best_idx].reshape(1, -1)
                fval = prev_vals[best_idx].item() * self.problem.sense.value

        return sol, fval
