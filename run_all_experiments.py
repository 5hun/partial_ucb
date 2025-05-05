import itertools as it
from pathlib import Path

import tomli_w
from tqdm import tqdm

import main


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

other_settings = {
    "seed_0": {
        "seed": 0,
    },
    "seed_1": {
        "seed": 1,
    },
    "seed_2": {
        "seed": 2,
    },
}

for p_key, m_key, o_key in tqdm(
    it.product(problems.keys(), methods.keys(), other_settings.keys())
):
    p_stg = problems[p_key]
    m_stg = methods[m_key]
    o_stg = other_settings[o_key]

    tmp_stg = base_settings.copy()
    tmp_stg.update(p_stg)
    tmp_stg.update(m_stg)
    tmp_stg.update(o_stg)

    tmp_output_dir = base_output_dir / p_key / m_key / o_key

    tmp_stg["output_dir"] = str(tmp_output_dir)

    tmp_output_dir.mkdir(parents=True, exist_ok=True)

    stg_file = tmp_output_dir / "settings.toml"
    with open(stg_file, "wb") as fout:
        tomli_w.dump(tmp_stg, fout)

    result_file = tmp_output_dir / "results.json"
    if result_file.exists():
        print(f"Result file {result_file} already exists. Skipping.")
        continue

    main.main(tmp_stg)
