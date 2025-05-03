import argparse
from pathlib import Path
import random
import tomllib as tml
import shutil
import json

import numpy as np
import torch

import util
import functions
import ackley
import query_algorithm

FUNCTIONS = {
    "ackley": ackley.get_ackley,
}


def set_random_seed(seed: int):
    r"""Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as fin:
        config = tml.load(fin)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.toml")

    set_random_seed(config["seed"])

    method_name = config["method"]
    method_config = config["method_config"]

    function_name = config["function"]
    function_config = config["function_config"]
    fun = FUNCTIONS[function_name](**function_config)

    assert method_name == "partial-ucb", "Currently only partial-ucb is supported"

    method = query_algorithm.PartialUCB(fun=fun, **method_config)

    # Get initial samples
    initial_samples = util.uniform_sampling(config["num_initial_samples"], fun.bounds)
    print(f"{initial_samples=}")
    initial_result = fun.eval(initial_samples)
    data = {}
    for nm, func in fun.name2func.items():
        if func.is_known:
            continue
        tmp_input = fun.get_input_tensor(nm, initial_result, initial_samples)
        print(f"{tmp_input.shape=}")
        print(f"{initial_result[nm].shape=}")
        print(f"{initial_result=}")
        assert tmp_input.shape == (initial_samples.shape[0], func.in_ndim)
        assert initial_result[nm].shape == (initial_samples.shape[0], func.out_ndim)
        data[nm] = (tmp_input, initial_result[nm])

    sol, est_val = method.get_solution(data)
    real_res = fun.eval(sol)
    real_val = real_res[fun.get_output_node()].item()
    results = [
        {
            "iter": 0,
            "estimated_solution": sol.tolist(),
            "estimated_value": est_val,
            "real_value": real_val,
        }
    ]

    with open(output_dir / "results_itermediate.json", "w") as fout:
        json.dump(results, fout, indent=4)

    num_iter = config["num_iter"]
    for i in range(num_iter):
        print(f"Iteration {i + 1}/{num_iter}")
        query = method.step(data)
        res = fun.eval_sub(query.query_function, query.query_input)

        data_input = data[query.query_function][0]
        data_output = data[query.query_function][1]
        data[query.query_function] = (
            torch.cat((data_input, query.query_input), dim=0),
            torch.cat((data_output, res), dim=0),
        )

        new_sol, new_est = method.get_solution(data)

        real_res = fun.eval(new_sol)
        real_val = real_res[fun.get_output_node()].item()
        results.append(
            {
                "iter": i + 1,
                "estimated_solution": new_sol.tolist(),
                "estimated_value": new_est,
                "real_value": real_val,
            }
        )

        with open(output_dir / "results_itermediate.json", "w") as fout:
            json.dump(results, fout, indent=4)

    with open(output_dir / "results.json", "w") as fout:
        json.dump(results, fout, indent=4)
