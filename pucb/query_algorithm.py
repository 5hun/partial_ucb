from dataclasses import dataclass
import logging

from jaxtyping import Float
import torch
from torch import Tensor

import botorch

from .problem import FunctionNetwork, Function, Problem, ObjectiveSense
from . import gp, optimize, util


@dataclass(frozen=True)
class QueryResponse:
    function_name: str
    input: Tensor
    info: dict


class QueryAlgorithm:
    def step(
        self, data: dict[str, tuple[Tensor, Tensor]], budget: int
    ) -> QueryResponse | None:
        raise NotImplementedError()

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        raise NotImplementedError()


class UCBCalculator:
    def __init__(
        self,
        problem: Problem,
        alpha: float,
        models: dict[str, botorch.models.SingleTaskGP],
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.models = models

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
            func = self.fun.functions[funcname]
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
        func = self.fun.functions[func_name]
        if func.is_known:
            res = self.fun.eval_sub(func_name, y)
        else:
            noise_offset = self.problem.bounds.shape[1]
            x_idx, x_idx_end = self.noise_info[node]
            x_idx += noise_offset
            x_idx_end += noise_offset
            noise = self.alpha * x[:, x_idx:x_idx_end]
            res = gp.ucb(self.models[node], y, noise)

        cache[node] = res

    def eval(self, x: Tensor) -> dict[str, Tensor]:
        cache = {}
        res_node = self.fun.get_output_node()
        self._eval_node(res_node, cache, x)
        return cache

    def __call__(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n 1"]:
        res = self.eval(x)
        return res[self.fun.get_output_node()]


class PartialUCB(QueryAlgorithm):
    r"""
    Partial-UCB strategy

    Parameters
    ----------
    problem : Problem
        The optimization problem to solve.
    alpha : float
        The exploration parameter for UCB.
    warm_start_model : bool
        Whether to warm start the GP model with the previous state when hyperparameter tuning.
    train_yvar : float, optional
        The estimated variance of the observation noise for training the GP model.
        Default is 1e-5.
    logger : logging.Logger, optional
        Logger for debugging and information messages. If None, a default logger is used.

    """

    def __init__(
        self,
        problem: Problem,
        alpha: float,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
        logger: logging.Logger | None = None,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        self.previous_sols = None
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.models = None

    def _optimize_UCBCalculator(
        self, models: dict[str, botorch.models.SingleTaskGP]
    ) -> tuple[Tensor, Tensor, float]:
        ucb_calc = UCBCalculator(self.problem, self.alpha, models)
        res = optimize.optimize(
            fun=ucb_calc,
            bounds=ucb_calc.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        self.logger.debug(f"UCB Optimization result: {res}")
        # assert res.success
        x_org = torch.tensor(res.x).reshape(1, -1)
        x = x_org[:, : self.problem.bounds.shape[1]]
        noise = x_org[:, self.problem.bounds.shape[1] :]
        fval = res.fun
        return x, noise, fval

    def step(
        self, data: dict[str, tuple[Tensor, Tensor]], budget: int
    ) -> QueryResponse:
        self.models = {
            nm: gp.fit_gp_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.models[nm].state_dict()
                    if self.models is not None and self.warm_start_model
                    else None
                ),
            )
            for nm, (x, y) in data.items()
            if nm != "__full__"
        }
        result, _noise, _result_fval = self._optimize_UCBCalculator(self.models)
        self.logger.debug(f"UCB Optimization result: {result=}")

        functions = self.fun.functions.copy()
        for nm in self.models:
            func = functions[nm]
            functions[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(self.models[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = FunctionNetwork(functions, self.fun.dag)

        eval_cache = tmp_fun.eval(result)
        res_node = tmp_fun.get_output_node()

        for nm in self.fun.functions:
            func = self.fun.functions[nm]
            if func.is_known:
                continue
            tmp_res = eval_cache[nm]
            tmp_res.retain_grad()
            assert tmp_res.requires_grad
            assert tmp_res.grad_fn is not None

        eval_cache[res_node].backward()

        best_r = float("-inf")
        best_response = None
        for nm in self.fun.functions:
            func = self.fun.functions[nm]
            if func.is_known:
                continue
            assert func.cost is not None
            if func.cost > budget:
                continue
            tmp_res = eval_cache[nm]
            tmp_grad = tmp_res.grad

            assert tmp_grad is not None
            assert tmp_grad.shape == tmp_res.shape
            tmp_input = tmp_fun.get_input_tensor(nm, eval_cache, result)
            cov = self.models[nm].posterior(tmp_input).covariance_matrix  # type: ignore
            r = (tmp_grad @ (cov @ tmp_grad)).item() / func.cost
            assert r >= 0
            self.logger.debug(f"Partial UCB: {nm}: {tmp_grad=}, {cov=}, {r=}")
            if r > best_r:
                best_r = r
                best_response = QueryResponse(
                    function_name=nm,
                    input=tmp_input.detach(),
                    info={"acquisition_value": r},
                )
        assert best_response is not None
        return best_response

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        self.models = {
            nm: gp.fit_gp_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.models[nm].state_dict()
                    if self.models is not None and self.warm_start_model
                    else None
                ),
            )
            for nm, (x, y) in data.items()
            if nm != "__full__"
        }
        functions = self.fun.functions.copy()
        for nm in self.models:
            func = functions[nm]
            functions[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(self.models[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = FunctionNetwork(functions, self.fun.dag)
        res = optimize.optimize(
            tmp_fun,
            self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        self.logger.debug(f"Posterior Mean Optimization result: {res}")
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
        self.logger.debug(
            f"Posterior Mean Optimization (using previous sols): {sol=}, {fval=}"
        )

        return sol, fval


class FunctionNetworkUCB(QueryAlgorithm):
    def __init__(
        self,
        problem: Problem,
        alpha: float,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
        logger: logging.Logger | None = None,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.alpha = alpha
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        self.previous_sols = None
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.models = None

    def _optimize_UCBCalculator(
        self, models: dict[str, botorch.models.SingleTaskGP]
    ) -> tuple[Tensor, Tensor, float]:
        ucb_calc = UCBCalculator(self.problem, self.alpha, models)
        res = optimize.optimize(
            fun=ucb_calc,
            bounds=ucb_calc.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        self.logger.debug(f"UCB Optimization result: {res}")
        # assert res.success
        x_org = torch.tensor(res.x).reshape(1, -1)
        x = x_org[:, : self.problem.bounds.shape[1]]
        noise = x_org[:, self.problem.bounds.shape[1] :]
        fval = res.fun
        return x, noise, fval

    def step(
        self, data: dict[str, tuple[Tensor, Tensor]], budget: int
    ) -> QueryResponse | None:
        if budget < self.problem.obj.total_cost():
            return None
        self.models = {
            nm: gp.fit_gp_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.models[nm].state_dict()
                    if self.warm_start_model and self.models is not None
                    else None
                ),
            )
            for nm, (x, y) in data.items()
            if nm != "__full__"
        }
        result, _noise, _result_fval = self._optimize_UCBCalculator(self.models)
        self.logger.debug(f"UCB Optimization result: {result=}")

        return QueryResponse(
            function_name="__full__",
            input=result,
            info={
                "acquisition_value": _result_fval,
            },
        )

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        self.models = {
            nm: gp.fit_gp_model(
                x,
                y,
                train_yvar=self.train_yvar,
                state_dict=(
                    self.models[nm].state_dict()
                    if self.models is not None and self.warm_start_model
                    else None
                ),
            )
            for nm, (x, y) in data.items()
            if nm != "__full__"
        }
        functions = self.fun.functions.copy()
        for nm in self.models:
            func = functions[nm]
            functions[nm] = Function(
                is_known=True,
                func=gp.ExpectationFunction(self.models[nm]),
                in_ndim=func.in_ndim,
                out_ndim=func.out_ndim,
            )
        tmp_fun = FunctionNetwork(functions, self.fun.dag)
        res = optimize.optimize(
            tmp_fun,
            self.problem.bounds,
            num_initial_samples=100,
            sense=self.problem.sense,
        )
        self.logger.debug(f"Posterior Mean Optimization result: {res}")
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
        self.logger.debug(
            f"Posterior Mean Optimization (using previous sols): {sol=}, {fval=}"
        )

        return sol, fval


class Random(QueryAlgorithm):
    def __init__(self, problem: Problem, logger: logging.Logger | None = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.problem = problem

    def step(
        self, data: dict[str, tuple[Tensor, Tensor]], budget: int
    ) -> QueryResponse | None:
        if budget < self.problem.obj.total_cost():
            return None
        new_input = util.uniform_sampling(1, self.problem.bounds)
        return QueryResponse(function_name="__full__", input=new_input, info={})

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        train_X, train_Y = data["__full__"]
        train_Y = train_Y * self.problem.sense.value
        best_idx = torch.argmin(train_Y)
        best_sol = train_X[best_idx].reshape(1, -1)
        best_val = train_Y[best_idx].item() * self.problem.sense.value
        return best_sol, best_val


class FullUCB(QueryAlgorithm):
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

    def step(
        self, data: dict[str, tuple[Tensor, Tensor]], budget: int
    ) -> QueryResponse | None:
        if budget < self.problem.obj.total_cost():
            return None
        train_X, train_Y = data["__full__"]
        self.model = gp.fit_gp_model(
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
        return QueryResponse(
            function_name="__full__",
            input=x,
            info={"acquisition_value": res.fun},
        )

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        train_X, train_Y = data["__full__"]
        self.model = gp.fit_gp_model(
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


class FullLogEI(QueryAlgorithm):
    def __init__(
        self,
        problem: Problem,
        warm_start_model: bool,
        train_yvar: float = 1e-5,
        logger: logging.Logger | None = None,
    ):
        self.problem = problem
        self.fun = problem.obj
        self.train_yvar = train_yvar
        self.warm_start_model = warm_start_model
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.model = None

    def step(self, data: dict, budget: int) -> QueryResponse | None:
        if budget < self.problem.obj.total_cost():
            return None
        train_X, train_Y = data["__full__"]
        self.model = gp.fit_gp_model(
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
        # assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        return QueryResponse(
            function_name="__full__",
            input=x,
            info={"acquisition_value": res.fun},
        )

    def get_solution(
        self, data: dict[str, tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, float]:
        train_X, train_Y = data["__full__"]
        self.model = gp.fit_gp_model(
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
        # assert res.success
        x = torch.tensor(res.x).reshape(1, -1)
        fval = res.fun
        return x, fval
