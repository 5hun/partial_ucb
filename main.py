import argparse
from pathlib import Path
import random
import tomllib as tml
import shutil
import json
import logging
import uuid

import numpy as np
import torch
from tqdm import tqdm
import tomli_w

import util
import functions
import benchmarks.ackley as ackley
import benchmarks.norm as norm
import query_algorithm

FUNCTIONS = {"ackley": ackley.get_ackley, "norm": norm.get_norm}

METHODS = {
    "partial-ucb": query_algorithm.PartialUCB,
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


def main_loop_partial(
    method: query_algorithm.PartialQueryAlgorithm,
    problem: functions.Problem,
    config: dict,
    logger: logging.Logger,
):
    output_dir = Path(config["output_dir"])

    # Get initial samples
    initial_samples = util.uniform_sampling(
        config["num_initial_samples"], problem.bounds
    )
    logger.debug(f"Generated {config['num_initial_samples']} initial samples")

    initial_result = problem.obj.eval(initial_samples)
    data = {}
    for nm, func in problem.obj.name2func.items():
        if func.is_known:
            continue
        tmp_input = problem.obj.get_input_tensor(nm, initial_result, initial_samples)
        assert tmp_input.shape == (initial_samples.shape[0], func.in_ndim)
        assert initial_result[nm].shape == (initial_samples.shape[0], func.out_ndim)
        data[nm] = (tmp_input, initial_result[nm])
    logger.debug(f"Initial data collected for {list(data.keys())}")

    sol, est_val = method.get_solution(data)
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
        },
        "iter": [],
    }
    logger.debug("Initial results computed")

    with open(output_dir / "results_itermediate.json", "w") as fout:
        json.dump(results, fout, indent=4)
    logger.debug("Initial results saved to intermediate file")

    num_iter = config["num_iter"]
    for i in tqdm(range(num_iter), desc="Iterations"):
        query = method.step(data)
        res = problem.obj.eval_sub(query.query_function, query.query_input)

        data_input = data[query.query_function][0]
        data_output = data[query.query_function][1]
        data[query.query_function] = (
            torch.cat((data_input, query.query_input), dim=0),
            torch.cat((data_output, res), dim=0),
        )
        logger.debug(
            f"Iteration {i + 1}: Data updated for function {query.query_function}"
        )

        new_sol, new_est = method.get_solution(data)
        real_val = problem.obj(new_sol).item()

        query_info = {
            "function": "__full__",
            "input": query.query_input.tolist()[0],
        }
        if "r" in query.info:
            query_info["acquisition_value"] = query.info["r"]
        results["iter"].append(
            {
                "iter": i + 1,
                "estimated_solution": new_sol.tolist()[0],
                "estimated_objective": new_est,
                "true_objective": real_val,
                "query": query_info,
            }
        )
        logger.debug(f"Iteration {i + 1}: Results updated")

        with open(output_dir / "results_itermediate.json", "w") as fout:
            json.dump(results, fout, indent=4)
        logger.debug(f"Iteration {i + 1}: Results saved to intermediate file")

    with open(output_dir / "results.json", "w") as fout:
        json.dump(results, fout, indent=4)
    logger.debug("Final results saved")


def main_loop_full(
    method: query_algorithm.FullQueryAlgorithm,
    problem: functions.Problem,
    config: dict,
    logger: logging.Logger,
):
    output_dir = Path(config["output_dir"])
    # Get initial samples
    train_X = util.uniform_sampling(config["num_initial_samples"], problem.bounds)
    logger.debug(f"Generated {config['num_initial_samples']} initial samples")

    train_Y = problem.obj(train_X)

    sol, est_val = method.get_solution(train_X, train_Y)

    real_val = problem.obj(sol).item()
    results = {
        "init": {
            "initial_samples": {
                "__full__": {
                    "input": train_X.tolist(),
                    "output": train_Y.tolist(),
                }
            },
            "estimated_solution": sol.tolist()[0],
            "estimated_objective": est_val,
            "true_objective": real_val,
        },
        "iter": [],
    }
    logger.debug("Initial results computed")

    with open(output_dir / "results_itermediate.json", "w") as fout:
        json.dump(results, fout, indent=4)
    logger.debug("Initial results saved to intermediate file")

    num_iter = config["num_iter"]
    for i in tqdm(range(num_iter), desc="Iterations"):
        query = method.step(train_X, train_Y)
        res = problem.obj(query.query_input)

        train_X = torch.cat((train_X, query.query_input), dim=0)
        train_Y = torch.cat((train_Y, res), dim=0)
        logger.debug(f"Iteration {i + 1}: Data updated")

        new_sol, new_est = method.get_solution(train_X, train_Y)
        real_val = problem.obj(new_sol).item()

        query_info = {
            "function": "__full__",
            "input": query.query_input.tolist()[0],
        }
        if "r" in query.info:
            query_info["acquisition_value"] = query.info["r"]
        results["iter"].append(
            {
                "iter": i + 1,
                "estimated_solution": new_sol.tolist()[0],
                "estimated_objective": new_est,
                "true_objective": real_val,
                "query": query_info,
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

    method: (
        query_algorithm.PartialQueryAlgorithm | query_algorithm.FullQueryAlgorithm
    ) = METHODS[method_name](problem=problem, logger=logger, **method_config)
    logger.debug(f"Method {method_name} initialized")

    if isinstance(method, query_algorithm.PartialQueryAlgorithm):
        main_loop_partial(method, problem, config, logger)
    else:
        assert isinstance(method, query_algorithm.FullQueryAlgorithm)
        main_loop_full(method, problem, config, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as fin:
        config = tml.load(fin)

    main(config)
