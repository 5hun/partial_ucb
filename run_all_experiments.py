import itertools as it
from pathlib import Path
import json
import multiprocessing as mp
from functools import partial

import tomli_w
from tqdm import tqdm
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns

import main
from joblib import Parallel, delayed


def process_experiment(
    base_settings: dict[str, str | int | float],
    problems: dict[str, dict[str, str | int | float]],
    methods: dict[str, dict[str, str | int | float]],
    base_output_dir: Path,
    combo: tuple[str, str, int],
) -> list[dict[str, str | int | float]]:
    p_key, m_key, seed = combo
    p_stg = problems[p_key]
    m_stg = methods[m_key]

    tmp_stg = base_settings.copy()
    tmp_stg.update(p_stg)
    tmp_stg.update(m_stg)
    tmp_stg.update({"seed": seed})

    tmp_output_dir = base_output_dir / p_key / m_key / f"seed_{seed}"
    tmp_stg["output_dir"] = str(tmp_output_dir)
    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    stg_file = tmp_output_dir / "settings.toml"
    with open(stg_file, "wb") as fout:
        tomli_w.dump(tmp_stg, fout)

    result_file = tmp_output_dir / "results.json"
    if not result_file.exists():
        main.main(tmp_stg)
    else:
        print(f"Result file {result_file} already exists. Skipping.")

    with open(result_file, "r") as fin:
        results = json.load(fin)

    exp_results = []
    init_res = results["init"]
    exp_results.append(
        {
            "problem": p_key,
            "method": m_key,
            "seed": seed,
            "iter": 0,
            "cost": init_res["initial_cost"],
            "obj": init_res["true_objective"],
        }
    )

    for i, res in enumerate(results["iter"]):
        exp_results.append(
            {
                "problem": p_key,
                "method": m_key,
                "seed": seed,
                "iter": i + 1,
                "cost": res["cost"],
                "obj": res["true_objective"],
            }
        )

    return exp_results


def create_all_plots(
    df: pl.DataFrame, plot_dir: Path, zero_init_cost: bool = True
) -> None:
    """Create all plots for the experiment results"""
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Add cumulative cost column
    df = df.with_columns(
        pl.col("cost")
        .cum_sum()
        .over(["problem", "method", "seed"])
        .alias("cumulative_cost")
        .cast(pl.Int64)
    )

    if zero_init_cost:
        df = df.with_columns(
            (
                pl.col("cumulative_cost")
                - pl.first("cumulative_cost").over(["problem", "method", "seed"])
            )
            .alias("cumulative_cost")
            .cast(pl.Int64)
        )

    for p in tqdm(sorted(df["problem"].unique()), desc="Plotting"):
        sub_df = df.filter(pl.col("problem") == p)

        # Prepare data for cost vs obj plot
        sub_df2 = sub_df.select(
            pl.col("method"), pl.col("seed"), pl.col("cumulative_cost"), pl.col("obj")
        )

        df_ranges = (
            sub_df2.group_by(["method", "seed"])
            .agg(
                [
                    pl.min("cumulative_cost").alias("start"),
                    (pl.max("cumulative_cost") + 1).alias("end"),
                ]
            )
            .with_columns(pl.int_ranges("start", "end").alias("cumulative_cost"))
            .explode("cumulative_cost")
            .select("method", "seed", "cumulative_cost")
        )

        # Join and forward fill
        filled_data = (
            df_ranges.join(
                sub_df2, on=["method", "seed", "cumulative_cost"], how="left"
            )
            .sort(["method", "seed", "cumulative_cost"])
            .with_columns(pl.col("obj").forward_fill())
        )

        # Plot 1: cost vs obj
        sns.relplot(
            data=filled_data.to_pandas(),
            x="cumulative_cost",
            y="obj",
            hue="method",
            kind="line",
            aspect=2,
        )
        plt.title(f"Problem {p}")
        plt.xlabel("Cumulative Cost")
        plt.ylabel("Objective Value")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{p}_cost_vs_obj.png")
        plt.close()

        # Plot 2: iter vs obj
        sns.relplot(
            data=sub_df.to_pandas(),
            x="iter",
            y="obj",
            hue="method",
            kind="line",
            aspect=2,
        )
        plt.title(f"Problem {p}")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{p}_iter_vs_obj.png")
        plt.close()

        # Plot 3: iter vs cost
        sns.relplot(
            data=sub_df.to_pandas(),
            x="iter",
            y="cumulative_cost",
            hue="method",
            kind="line",
            aspect=2,
        )
        plt.title(f"Problem {p}")
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Cost")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{p}_iter_vs_cost.png")
        plt.close()


if __name__ == "__main__":
    base_output_dir = Path("output/experiments")

    num_parallel = 6

    base_settings = {
        "log_level": "DEBUG",
        "max_iter": 700,
        "budget": 700,
        "ignore_initial_cost": True,
    }

    problems = {
        # "ackley_2d": {
        #     "function": "ackley",
        #     "function_config": {"ndim": 2, "cost1": 1, "cost2": 1},
        #     "num_initial_samples": 5,
        # },
        "ackley_6d_1_1": {
            "function": "ackley",
            "function_config": {"ndim": 6, "cost1": 1, "cost2": 1},
            "num_initial_samples": 13,
        },
        "ackley_6d_1_9": {
            "function": "ackley",
            "function_config": {"ndim": 6, "cost1": 1, "cost2": 9},
            "num_initial_samples": 13,
        },
        "ackley_6d_1_49": {
            "function": "ackley",
            "function_config": {"ndim": 6, "cost1": 1, "cost2": 49},
            "num_initial_samples": 13,
        },
        "pharma": {
            "function": "pharma",
            "function_config": {},
            "num_initial_samples": 9,
        },
        "norm_2d": {
            "function": "norm",
            "function_config": {"ndim": 2, "p": 2},
            "num_initial_samples": 5,
        },
    }

    methods = {
        # "random": {"method": "random", "method_config": {}},
        "partial-ucb_1": {
            "method": "partial-ucb",
            "method_config": {
                "alpha": 1.0,
                "train_yvar": 1e-5,
                "warm_start_model": True,
            },
        },
        # "partial-ucb_2": {"method": "partial-ucb", "method_config": {"alpha": 2.0}},
        "full-ucb_1": {
            "method": "full-ucb",
            "method_config": {
                "alpha": 1.0,
                "warm_start_model": True,
                "train_yvar": 1e-5,
            },
        },
        # "full-ucb_2": {"method": "full-ucb", "method_config": {"alpha": 2.0}},
        "full-logei": {
            "method": "full-logei",
            "method_config": {"warm_start_model": True, "train_yvar": 1e-5},
        },
    }

    # Create list of all parameter combinations
    param_combos = list(it.product(problems.keys(), methods.keys(), range(30)))

    # Prepare worker function with fixed parameters
    worker_func = partial(
        process_experiment, base_settings, problems, methods, base_output_dir
    )

    # Use joblib for parallel processing
    results = Parallel(n_jobs=num_parallel, verbose=10)(
        delayed(worker_func)(combo) for combo in param_combos
    )

    # Flatten results list
    df = [item for sublist in results for item in sublist]  # type: ignore

    df = pl.DataFrame(df)
    df.write_csv(base_output_dir / "summary.csv")

    create_all_plots(df, base_output_dir / "plots")
