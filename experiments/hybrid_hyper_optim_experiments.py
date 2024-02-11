import os
import json
import hybrid_hypercut_greedy as hhg
import time
import pickle
import cotengra as ctg
import opt_einsum as oe
from pathlib import Path
from hybrid_experiments import (
    get_remaining as hybrid_remaining,
    result_dir,
    get_filename,
    serialize_object,
)


def store_path(
    circuit_file,
    max_seconds,
    instance,
    imbalance,
    weight_function,
    hyper_params,
    search_time,
    path,
    opt_cost,
    trials,
    time_result,
):
    Path(result_dir).mkdir(parents=False, exist_ok=True)
    filename = get_filename(
        circuit_file,
        max_seconds,
        "optimized",
        weight_function,
        hyper_params,
        instance,
    )

    data = {
        "circuit_file": circuit_file,
        "instance": instance,
        "max_seconds": max_seconds,
        "imbalance": imbalance,
        "weight_function": weight_function,
        "hyper_params": hyper_params,
        "search_time": search_time,
        "contract_path": path,
        "opt_cost": str(opt_cost),
        "trials": trials,
        "time_result": time_result,
    }

    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data_list = json.load(f)
                if isinstance(data_list, list):
                    data_list.append(data)
                else:
                    data_list = [data]
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")
                data_list = [data]
    else:
        data_list = [data]

    with open(filename, "w") as f:
        json.dump(data_list, f, default=serialize_object)


num_samples = 10

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
circuit_files = [
    "circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p",
    "TN_1688_all.p",
    "TN_1688_remaining_steps_1000.p",
    "TN_1688_remaining_steps_2000.p",
]


circuit_file = circuit_files[task_id]
with open(f"dataset/{circuit_file}", "rb") as f:
    task = pickle.load(f)

instance = 0
eq = task[0][instance][0]


def tree_to_path_info(tree: ctg.ContractionTree):
    return oe.contract_path(
        eq, *tree.get_shapes(), optimize=tree.get_path(), shapes=True
    )


shapes = task[0][instance][1]
size_dict = task[0][instance][2]
splitted = eq.split("->")
inputs: hhg.Inputs = [list(input) for input in splitted[0].split(",")]
output = list(splitted[1])
max_times = [300, 600, 1800]

weight_function = (0, 0, 5)
hyper_params = {
    "one_sided_output": True,
    "num_output_nodes": 1,
    "parts": 2,
}
optlib = "optuna"
imbalance = "optimized"


repeats = 5000000
for max_time in max_times:
    remaining = hybrid_remaining(
        circuit_file,
        max_time,
        imbalance,
        weight_function,
        hyper_params,
        instance,
        num_samples,
    )
    print(f"For {imbalance}, {weight_function}, {hyper_params} remaining: {remaining}")
    for i in range(remaining):
        optimizer = ctg.HyperOptimizer(
            max_repeats=repeats,
            methods=["hhg"],
            optlib=optlib,
            minimize="flops",
            parallel=False,
            progbar=True,
            max_time=max_time,
        )

        start = time.time()
        tree = optimizer.search(inputs, output, size_dict)
        end = time.time()

        search_time = end - start
        print("Search Time:", search_time)

        path, path_info = tree_to_path_info(tree)
        trials = optimizer.get_trials(sort="flops")

        print("Opt cost", path_info.opt_cost)
        print("Trials", len(trials))

        print(trials)
        best_imbalance = trials[0][4]

        print(best_imbalance)
        store_path(
            circuit_file,
            max_time,
            instance,
            best_imbalance["imbalance"],
            weight_function,
            hyper_params,
            search_time,
            path_info.path,
            path_info.opt_cost,
            len(trials),
            [],
        )
