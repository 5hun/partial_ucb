from dataclasses import dataclass
import logging

from jaxtyping import Float
import torch
from torch import Tensor

import botorch

from functions import DAGFunction, Function, Problem, ObjectiveSense
import gp
import optimize
import util


@dataclass(frozen=True)
class PartialQueryResponse:
    query_function: str
    query_input: Tensor
    info: dict


class PartialQueryAlgorithm:
    def step(self, data: dict[str, tuple[Tensor, Tensor]]) -> PartialQueryResponse:
        raise NotImplementedError()

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        raise NotImplementedError()


@dataclass(frozen=True)
class FullQueryResponse:
    query_input: Tensor
    info: dict


class FullQueryAlgorithm:
    def step(self, train_X: Tensor, train_Y: Tensor) -> FullQueryResponse:
        raise NotImplementedError()

    def get_solution(self, train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, float]:
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


class PartialUCB(PartialQueryAlgorithm):
    def __init__(
        self,
        problem: Problem,
        alpha: float,
        logger: logging.Logger,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        self.previous_sols = None
        self.logger = logger
        self.mods = None

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
        self.logger.debug(f"Optimization result: {res}")
        # assert res.success
        x_org = torch.tensor(res.x).reshape(1, -1)
        x = x_org[:, : self.problem.bounds.shape[1]]
        noise = x_org[:, self.problem.bounds.shape[1] :]
        fval = res.fun
        return x, noise, fval

    def step(self, data: dict[str, tuple[Tensor, Tensor]]) -> PartialQueryResponse:
        self.mods = {
            nm: gp.get_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.mods[nm].state_dict()
                    if self.mods is not None and self.warm_start_model
                    else None
                ),
            )
            for nm, (x, y) in data.items()
        }
        result, _noise, _result_fval = self._optimize_dagucb(self.mods)
        self.logger.debug(f"Optimize DAGUCB result: {result=}")

        name2func = self.fun.name2func.copy()
        for nm in self.mods:
            func = name2func[nm]
            name2func[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(self.mods[nm]),
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
            cov = self.mods[nm].posterior(tmp_input).covariance_matrix  # type: ignore
            r = (tmp_grad @ (cov @ tmp_grad)).item() / func.cost
            assert r >= 0
            self.logger.debug(f"Partial UCB: {nm}: {tmp_grad=}, {cov=}, {r=}")
            if r > best_r:
                best_r = r
                best_response = PartialQueryResponse(
                    query_function=nm, query_input=tmp_input.detach(), info={"r": r}
                )
        assert best_response is not None
        return best_response

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        self.mods = {
            nm: gp.get_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.mods[nm].state_dict()
                    if self.mods is not None and self.warm_start_model
                    else None
                ),
            )
            for nm, (x, y) in data.items()
        }
        name2func = self.fun.name2func.copy()
        for nm in self.mods:
            func = name2func[nm]
            name2func[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(self.mods[nm]),
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
        self.logger.debug(f"Optimization result: {res}")
        # assert res.success
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


class Random(FullQueryAlgorithm):
    def __init__(self, problem: Problem, logger: logging.Logger):
        self.problem = problem

    def step(self, train_X: Tensor, train_Y: Tensor) -> FullQueryResponse:
        new_input = util.uniform_sampling(1, self.problem.bounds)
        return FullQueryResponse(query_input=new_input, info={})

    def get_solution(self, train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, float]:
        train_Y = train_Y * self.problem.sense.value
        best_idx = torch.argmin(train_Y)
        best_sol = train_X[best_idx].reshape(1, -1)
        best_val = train_Y[best_idx].item() * self.problem.sense.value
        return best_sol, best_val


class FullUCB(FullQueryAlgorithm):
    def __init__(
        self,
        problem: Problem,
        alpha: float,
        logger: logging.Logger,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        self.logger = logger
        self.model = None

    def step(self, train_X: Tensor, train_Y: Tensor) -> FullQueryResponse:
        self.model = gp.get_model(
            train_X,
            train_Y,
            train_yvar=self.train_yvar,
            state_dict=(
                self.model.state_dict()
                if self.warm_start_model and self.model is not None
                else None
            ),
        )
        res = optimize.optimize(
            fun=gp.UCBFunction(
                self.model,
                self.alpha,
                lower=True if self.problem.sense == ObjectiveSense.MINIMIZE else False,
            ),
            bounds=self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        self.logger.debug(f"Optimize UCB result: {res}")
        assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        return FullQueryResponse(query_input=x, info={"r": res.fun})

    def get_solution(self, train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, float]:
        self.model = gp.get_model(
            train_X,
            train_Y,
            train_yvar=self.train_yvar,
            state_dict=(
                self.model.state_dict()
                if self.warm_start_model and self.model is not None
                else None
            ),
        )
        res = optimize.optimize(
            fun=gp.ExpectationFunction(self.model),
            bounds=self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        fval = res.fun
        return x, fval


class FullLogEI(FullQueryAlgorithm):
    def __init__(
        self,
        problem: Problem,
        logger: logging.Logger,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        self.logger = logger
        self.model = None

    def step(self, train_X: Tensor, train_Y: Tensor) -> FullQueryResponse:
        self.model = gp.get_model(
            train_X,
            train_Y,
            train_yvar=self.train_yvar,
            state_dict=(
                self.model.state_dict()
                if self.warm_start_model and self.model is not None
                else None
            ),
        )

        best_f = (
            train_Y.max()
            if self.problem.sense == ObjectiveSense.MAXIMIZE
            else train_Y.min()
        ).item()
        logei_acq = botorch.acquisition.LogExpectedImprovement(
            model=self.model,
            best_f=best_f,
            maximize=True if self.problem.sense == ObjectiveSense.MAXIMIZE else False,
        )
        res = optimize.optimize(
            fun=lambda x: logei_acq.forward(x.reshape(-1, 1, x.shape[-1])).reshape(
                -1, 1
            ),
            bounds=self.problem.bounds,
            num_initial_samples=100,
            sense=ObjectiveSense.MAXIMIZE,
        )
        self.logger.debug(f"Optimize LogEI result: {res}")
        assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        return FullQueryResponse(query_input=x, info={"r": res.fun})

    def get_solution(self, train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, float]:
        self.model = gp.get_model(
            train_X,
            train_Y,
            train_yvar=self.train_yvar,
            state_dict=(
                self.model.state_dict()
                if self.warm_start_model and self.model is not None
                else None
            ),
        )
        res = optimize.optimize(
            fun=gp.ExpectationFunction(self.model),
            bounds=self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        fval = res.fun
        return x, fval
