import argparse
import tomllib as tml
from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt


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
    print(f"{obj_vals=}, {est_vals=}, {acq_vals=}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(obj_vals, label="True Objective")
    axs[0].plot(est_vals, label="Estimated Objective")
    axs[0].set_title("Objective Values")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Objective Value")
    axs[0].legend()

    axs[1].plot(obj_vals - est_vals, label="Error")
    axs[1].set_title("Error")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Error")
    axs[1].legend()

    axs[2].plot(acq_vals, label="Acquisition Value")
    axs[2].set_title("Acquisition Values")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Acquisition Value")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(vis_dir / "objective_values.png")
    plt.close(fig)

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
        print(f"{nm=}, {query_inputs=}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "rb") as fin:
        config = tml.load(fin)

    visualize(config)
