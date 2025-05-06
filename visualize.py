import argparse
import tomllib as tml
from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt


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


def plot_objective_values(result: dict, vis_dir: Path) -> None:
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


def plot_query_inputs(result: dict, vis_dir: Path) -> None:
    names = sorted(
        set(x["query"]["function"] for x in result["iter"])
        | set(result["init"]["initial_samples"].keys())
    )
    for nm in names:
        # plot query inputs
        init_inputs = np.array(result["init"]["initial_samples"][nm]["input"])
        query_inputs = np.array(
            [
                x["query"]["input"]
                for x in result["iter"]
                if x["query"]["function"] == nm
            ]
        )
        dim = init_inputs.shape[1]
        if dim > 2:
            print(f"Skipping {nm} because it has more than 2 dimensions.")
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        if dim == 1:
            ax.plot(
                init_inputs, np.zeros_like(init_inputs), "o", label="Initial Inputs"
            )
            ax.plot(
                query_inputs, np.zeros_like(query_inputs), "o", label="Query Inputs"
            )
            for i, point in enumerate(query_inputs):
                ax.annotate(
                    str(i + 1),
                    (point, 0),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
        else:
            ax.plot(init_inputs[:, 0], init_inputs[:, 1], "o", label="Initial Inputs")
            ax.plot(query_inputs[:, 0], query_inputs[:, 1], "o", label="Query Inputs")
            for i, point in enumerate(query_inputs):
                ax.annotate(
                    str(i + 1),
                    (point[0], point[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
        ax.set_title(f"Query Inputs for {nm}")
        ax.legend()
        plt.savefig(vis_dir / f"{nm}_query_inputs.png")
        plt.close(fig)


def plot_proposed_solutions(result: dict, vis_dir: Path) -> None:
    solutions = [result["init"]["estimated_solution"]] + [
        x["estimated_solution"] for x in result["iter"]
    ]
    solutions = np.array(solutions)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(solutions[:, 0], solutions[:, 1], "o", label="Proposed Solutions")
    for i, point in enumerate(solutions):
        ax.annotate(
            str(i + 1),
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_title("Proposed Solutions")
    ax.legend()
    plt.savefig(vis_dir / "proposed_solutions.png")
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

    plot_objective_values(result, vis_dir)
    plot_query_inputs(result, vis_dir)
    if len(result["init"]["estimated_solution"]) == 2:
        plot_proposed_solutions(result, vis_dir)
    else:
        print(
            f"Skipping proposed solutions plot because the solution dimension is {len(result['init']['estimated_solution'])}."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as fin:
        config = tml.load(fin)

    visualize(config)
