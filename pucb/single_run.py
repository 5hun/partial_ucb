import argparse
from pathlib import Path
import random
import tomllib as tml
import json
import logging
import uuid
import time

import numpy as np
import torch
from tqdm import tqdm
import tomli_w
import matplotlib
from matplotlib import pyplot as plt

from . import util, functions, query_algorithm
from .test_functions import ackley, norm, pharma, freesolv3


FUNCTIONS = {
    "ackley": ackley.get_ackley,
    "norm": norm.get_norm,
    "pharma": pharma.get_pharma,
    "freesolv3": freesolv3.get_freesolv3,
}

METHODS = {
    "partial-ucb": query_algorithm.PartialUCB,
    "fn-ucb": query_algorithm.FNUCB,
    "full-ucb": query_algorithm.FullUCB,
    "full-logei": query_algorithm.FullLogEI,
    "random": query_algorithm.Random,
}


def set_random_seed(seed: int):
    r"""Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logging(output_dir: Path, log_level: str):
    """Set up logging configuration."""
    log_file = output_dir / "log.txt"

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper())

    # Configure logging: clear file if it exists, then set up handler
    if log_file.exists():
        log_file.unlink()

    # Create logger
    logger_name = f"pucb.run.{output_dir.name}.{uuid.uuid4()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Remove any existing handlers
    if logger.handlers:
        logger.handlers = []

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    logger.debug(f"Logging initialized at level {log_level}")

    return logger


def run_experiment(
    method: query_algorithm.QueryAlgorithm,
    problem: functions.Problem,
    config: dict,
    logger: logging.Logger,
):
    output_dir = Path(config["output_dir"])

    max_iter = config["max_iter"]
    budget = config["budget"]

    # Get initial samples
    initial_samples = util.uniform_sampling(
        config["num_initial_samples"], problem.bounds
    )
    logger.debug(f"Generated {config['num_initial_samples']} initial samples")

    initial_result = problem.obj.eval(initial_samples)
    initial_cost = 0
    data = {}
    for nm, func in problem.obj.name2func.items():
        if func.is_known:
            continue
        tmp_input = problem.obj.get_input_tensor(nm, initial_result, initial_samples)
        assert tmp_input.shape == (initial_samples.shape[0], func.in_ndim)
        assert initial_result[nm].shape == (initial_samples.shape[0], func.out_ndim)
        data[nm] = (tmp_input, initial_result[nm])
        initial_cost += problem.obj.get_cost(nm) * initial_samples.shape[0]

    result_name = problem.obj.get_output_node_name()
    data["__full__"] = (
        initial_samples,
        initial_result[result_name],
    )

    logger.debug(f"Initial data collected for {list(data.keys())}")

    if not config["ignore_initial_cost"]:
        budget -= initial_cost
    if budget < 0:
        raise ValueError(
            f"Budget {budget} is less than 0 after initial cost {initial_cost}"
        )

    t1 = time.time()
    sol, est_val = method.get_solution(data)
    get_solution_time = time.time() - t1
    real_val = problem.obj(sol).item()
    results = {
        "init": {
            "initial_samples": {
                nm: {
                    "input": data[nm][0].tolist(),
                    "output": data[nm][1].tolist(),
                }
                for nm in data
            },
            "estimated_solution": sol.tolist()[0],
            "estimated_objective": est_val,
            "true_objective": real_val,
            "initial_cost": initial_cost,
            "get_solution_time": get_solution_time,
        },
        "iter": [],
    }
    logger.debug("Initial results computed")

    with open(output_dir / "results_itermediate.json", "w") as fout:
        json.dump(results, fout, indent=4)
    logger.debug("Initial results saved to intermediate file")

    for i in tqdm(range(max_iter), desc="Iterations"):
        if len(problem.obj.available_funcs(budget)) == 0:
            logger.debug(
                f"No available functions to query with remaining budget: {budget}"
            )
            break

        t1 = time.time()
        query = method.step(data, budget)
        get_query_time = time.time() - t1
        if query is None:
            logger.debug(f"No query returned with remaining budget: {budget}")
            break
        elif query.function_name != "__full__":
            res = problem.obj.eval_sub(query.function_name, query.input)
            data_input = data[query.function_name][0]
            data_output = data[query.function_name][1]
            data[query.function_name] = (
                torch.cat((data_input, query.input), dim=0),
                torch.cat((data_output, res), dim=0),
            )
            tmp_cost = problem.obj.get_cost(query.function_name)
            assert budget >= tmp_cost
            budget -= tmp_cost
            logger.debug(f"Query evaluated: {query}, {res.tolist()[0]}")
            logger.debug(
                f"Iteration {i + 1}: Data updated for function {query.function_name}"
            )
            query_info = {
                "function": query.function_name,
                "input": query.input.tolist()[0],
                "result": {
                    query.function_name: {
                        "input": query.input.tolist()[0],
                        "output": res.tolist()[0],
                    }
                },
            }
            if "acquisition_value" in query.info:
                query_info["acquisition_value"] = query.info["acquisition_value"]
        else:
            assert query.function_name == "__full__"
            res = problem.obj.eval(query.input)

            query_info = {
                "function": query.function_name,
                "input": query.input.tolist()[0],
                "result": {},
            }

            for nm, func in problem.obj.name2func.items():
                if func.is_known:
                    continue
                tmp_input = problem.obj.get_input_tensor(nm, res, query.input)
                assert tmp_input.shape == (query.input.shape[0], func.in_ndim)
                assert res[nm].shape == (query.input.shape[0], func.out_ndim)
                data[nm] = (
                    torch.cat((data[nm][0], tmp_input), dim=0),
                    torch.cat((data[nm][1], res[nm]), dim=0),
                )
                query_info["result"][nm] = {
                    "input": tmp_input.tolist(),
                    "output": res[nm].tolist(),
                }

            result_name = problem.obj.get_output_node_name()
            data["__full__"] = (
                torch.cat((data["__full__"][0], query.input), dim=0),
                torch.cat((data["__full__"][1], res[result_name]), dim=0),
            )

            tmp_cost = problem.obj.total_cost()
            assert budget >= tmp_cost
            budget -= tmp_cost

            if "acquisition_value" in query.info:
                query_info["acquisition_value"] = query.info["acquisition_value"]

        t1 = time.time()
        new_sol, new_est = method.get_solution(data)
        get_solution_time = time.time() - t1
        real_val = problem.obj(new_sol).item()
        logger.debug(
            f"Iteration {i + 1}: Proposed solution's estimated value: {new_est}, real value: {real_val}"
        )

        results["iter"].append(
            {
                "iter": i + 1,
                "estimated_solution": new_sol.tolist()[0],
                "estimated_objective": new_est,
                "true_objective": real_val,
                "query": query_info,
                "cost": tmp_cost,
                "get_query_time": get_query_time,
                "get_solution_time": get_solution_time,
            }
        )
        logger.debug(f"Iteration {i + 1}: Results updated")

        with open(output_dir / "results_itermediate.json", "w") as fout:
            json.dump(results, fout, indent=4)
        logger.debug(f"Iteration {i + 1}: Results saved to intermediate file")

    with open(output_dir / "results.json", "w") as fout:
        json.dump(results, fout, indent=4)
    logger.debug("Final results saved")


def main(config: dict):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_level = config.get("log_level", "INFO")
    logger = setup_logging(output_dir, log_level)

    with open(output_dir / "config.toml", "wb") as fout:
        tomli_w.dump(config, fout)
    logger.debug(f"Config file copied to {output_dir / 'config.toml'}")

    set_random_seed(config["seed"])
    logger.debug(f"Random seed set to {config['seed']}")

    method_name = config["method"]
    method_config = config["method_config"]
    logger.debug(f"Method: {method_name}, Config: {method_config}")

    function_name = config["function"]
    function_config = config["function_config"]
    logger.debug(f"Function: {function_name}, Config: {function_config}")

    problem = FUNCTIONS[function_name](**function_config)

    assert method_name in METHODS, f"Method {method_name} is not supported"

    method: query_algorithm.QueryAlgorithm = METHODS[method_name](
        problem=problem, logger=logger, **method_config
    )
    logger.debug(f"Method {method_name} initialized")

    run_experiment(method, problem, config, logger)


def step_function_intepolate(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    x = [1, 3, 7]
    y = [0, 1, 2]
    =>
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [0, 0, 1, 1, 1, 1, 2]
    """
    x_exp = []
    y_exp = []
    for i in range(len(x) - 1):
        x_exp.append(x[i])
        y_exp.append(y[i])
        x_diff = x[i + 1] - x[i]
        if x_diff > 1:
            for j in range(1, x_diff):
                x_exp.append(x[i] + j)
                y_exp.append(y[i])
    x_exp.append(x[-1])
    y_exp.append(y[-1])
    return np.array(x_exp), np.array(y_exp)


def plot_objective_values(config: dict, result: dict, vis_dir: Path) -> None:
    obj_vals = np.array(
        [result["init"]["true_objective"]]
        + [x["true_objective"] for x in result["iter"]]
    )
    est_vals = np.array(
        [result["init"]["estimated_objective"]]
        + [x["estimated_objective"] for x in result["iter"]]
    )
    acq_vals = np.array(
        [float("nan")] + [x["query"]["acquisition_value"] for x in result["iter"]]
    )

    costs = np.array(
        [result["init"]["initial_cost"]] + [x["cost"] for x in result["iter"]]
    )
    if config["ignore_initial_cost"]:
        costs[0] = 0
    cum_costs = np.cumsum(costs)
    cum_costs2, obj_vals2 = step_function_intepolate(cum_costs, obj_vals)
    _, est_vals2 = step_function_intepolate(cum_costs, est_vals)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    axs[0].plot(obj_vals, label="True Objective")
    axs[0].plot(est_vals, label="Estimated Objective")
    axs[0].set_title("Objective Values")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Objective Value")
    axs[0].legend()

    # cum costs vs obj
    axs[1].plot(cum_costs2, obj_vals2, label="True Objective")
    axs[1].plot(cum_costs2, est_vals2, label="Estimated Objective")
    axs[1].set_title("Cumulative Cost vs Objective Values")
    axs[1].set_xlabel("Cumulative Cost")
    axs[1].set_ylabel("Objective Value")
    axs[1].legend()

    axs[2].plot(obj_vals - est_vals, label="Error")
    axs[2].set_title("Error")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Error")
    axs[2].legend()

    axs[3].plot(acq_vals, label="Acquisition Value")
    axs[3].set_title("Acquisition Values")
    axs[3].set_xlabel("Iteration")
    axs[3].set_ylabel("Acquisition Value")
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(vis_dir / "objective_values.png")
    plt.close(fig)


def visualize(config: dict) -> None:
    output_dir = Path(config["output_dir"])

    sol_file = output_dir / "results.json"
    if not sol_file.exists():
        sol_file = output_dir / "results_intermediate.json"
    if not sol_file.exists():
        raise FileNotFoundError(
            f"Solution files results.json and results_intermediate.json are not found."
        )

    with open(sol_file, "r") as fin:
        result = json.load(fin)

    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    matplotlib.use("Agg")
    plot_objective_values(config, result, vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as fin:
        config = tml.load(fin)

    main(config)
    visualize(config)
