import itertools as it
from pathlib import Path
import json

import tomli_w
from tqdm import tqdm
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns

base_output_dir = Path("output")

base_settings = {
    "log_level": "DEBUG",
    "num_initial_samples": 5,
    "num_iter": 20,
}

problems = {
    "ackley_2d": {"function": "ackley", "function_config": {"ndim": 2}},
    "ackley_6d": {"function": "ackley", "function_config": {"ndim": 6}},
    "norm_2d": {"function": "norm", "function_config": {"ndim": 2, "p": 2}},
}

methods = {
    "random": {"method": "random", "method_config": {}},
    "partial-ucb_1": {"method": "partial-ucb", "method_config": {"alpha": 1.0}},
    "partial-ucb_2": {"method": "partial-ucb", "method_config": {"alpha": 2.0}},
    "full-ucb_1": {"method": "full-ucb", "method_config": {"alpha": 1.0}},
    "full-ucb_2": {"method": "full-ucb", "method_config": {"alpha": 2.0}},
    "full-logei": {"method": "full-logei", "method_config": {}},
}


df = []

for p_key, m_key, seed in tqdm(it.product(problems.keys(), methods.keys(), range(3))):
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
        import main

        main.main(tmp_stg)
    else:
        print(f"Result file {result_file} already exists. Skipping.")

    with open(result_file, "r") as fin:
        results = json.load(fin)

    init_res = results["init"]
    df.append(
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
        df.append(
            {
                "problem": p_key,
                "method": m_key,
                "seed": seed,
                "iter": i + 1,
                "cost": res["cost"],
                "obj": res["true_objective"],
            }
        )

df = pl.DataFrame(df)
df.write_csv(base_output_dir / "summary.csv")

plot_dir = base_output_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
df = df.with_columns(
    pl.col("cost")
    .cum_sum()
    .over(["problem", "method", "seed"])
    .alias("cumulative_cost")
)
for p in sorted(df["problem"].unique()):
    sub_df = df.filter(pl.col("problem") == p)
    # plot cost vs obj, coloring by method
    sns.relplot(
        data=sub_df.to_pandas(),
        x="cumulative_cost",
        y="obj",
        hue="method",
        kind="line",
        # height=5,
        aspect=2,
        # estimator=None,
        # units="seed"
    )
    plt.title(f"Problem {p}")
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Objective Value")
    plt.legend(title="Method")
    plt.savefig(plot_dir / f"{p}_cost_vs_obj.png")
    plt.close()

    # plot Iter vs obj, coloring by method
    sns.relplot(
        data=sub_df.to_pandas(),
        x="iter",
        y="obj",
        hue="method",
        kind="line",
        # height=5,
        aspect=2,
        # estimator=None,
        # units="seed"
    )
    plt.title(f"Problem {p}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.legend(title="Method")
    plt.savefig(plot_dir / f"{p}_iter_vs_obj.png")
    plt.close()
    # plot Iter vs cost, coloring by method
    sns.relplot(
        data=sub_df.to_pandas(),
        x="iter",
        y="cumulative_cost",
        hue="method",
        kind="line",
        # height=5,
        aspect=2,
        # estimator=None,
        # units="seed"
    )
    plt.title(f"Problem {p}")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Cost")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{p}_iter_vs_cost.png")
    plt.close()
