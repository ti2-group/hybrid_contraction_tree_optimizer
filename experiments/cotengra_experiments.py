import json
import os
import pickle
import time
from pathlib import Path

import cotengra as ctg
import opt_einsum as oe

DEBUG = False

result_dir = "results/cotengra/"


def tree_to_path_info(tree: ctg.ContractionTree):
    return oe.contract_path(
        eq, *tree.get_shapes(), optimize=tree.get_path(), shapes=True
    )


def get_filename(circuit_file, instance, max_time, optlib, method):
    return f"{result_dir}{circuit_file}_{instance}_{max_time}_{optlib}_{method}.json"


def get_remaining(
    circuit_file,
    instance,
    max_time,
    optlib,
    method,
    target,
):
    filename = get_filename(
        circuit_file,
        instance,
        max_time,
        optlib,
        method,
    )
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data_list = json.load(f)
                return target - len(data_list)
            except json.JSONDecodeError:
                return target
    return target


def store_path(
    circuit_file,
    instance,
    max_time,
    optlib,
    method,
    search_time,
    trials,
    path,
    opt_cost,
):
    Path(result_dir).mkdir(parents=False, exist_ok=True)
    filename = get_filename(
        circuit_file,
        instance,
        max_time,
        optlib,
        method,
    )

    data = {
        "circuit_file": circuit_file,
        "instance": instance,
        "max_time": max_time,
        "optlib": optlib,
        "method": method,
        "search_time": search_time,
        "contract_path": path,
        "trials": trials,
        "opt_cost": str(opt_cost),
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
        json.dump(data_list, f)


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
circuit_files = [
    # "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_25_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_50_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_75_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_100_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_125_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_150_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_175_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_200_mean_conn_3.p",
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_225_mean_conn_3.p",
    "circuit_n53_m10_s0_e0_pABCDCDAB_simplified_baselines.p",
    "circuit_n53_m12_s0_e0_pABCDCDAB_simplified_baselines.p",
    "circuit_n53_m14_s0_e0_pABCDCDAB_simplified_baselines.p",
    "circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p",
    "TN_1688_all.p",
    "TN_1688_remaining_steps_1000.p",
    "TN_1688_remaining_steps_2000.p",
]


file_based_config = {
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_25_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [1, 10, 30],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_50_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [5, 10, 30, 60],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_75_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [5, 10, 60, 120],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_100_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_125_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_150_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_175_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_200_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_225_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        # "all_optlibs": True,
    },
    "circuit_n53_m10_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
    },
    "circuit_n53_m12_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
    },
    "circuit_n53_m14_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
    },
    "circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600, 10800],
        # "all_optlibs": True,
    },
    "TN_1688_all.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600, 10800, 21600],
        "optuna": True,
        # "all_optlibs": True,
    },
    "TN_1688_remaining_steps_1000.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600, 10800],
        "ablation": False,
    },
    "TN_1688_remaining_steps_2000.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600, 10800],
        "ablation": False,
    },
}


methods = ["greedy", "kahypar"]
mehotd_optlib = {
    "greedy": "random",
    "kahypar": "optuna",
}
optlibs = ["random", "optuna", "nevergrad"]

tasks = []
for circuit_file in circuit_files:
    if "all_optlibs" in file_based_config[circuit_file]:
        for optlib in optlibs:
            for method in methods:
                tasks.append((circuit_file, optlib, method))
    for method in methods:
        optlib = mehotd_optlib[method]
        tasks.append((circuit_file, optlib, method))


if __name__ == "__main__":
    print(len(tasks))
    circuit_file, optlib, method = tasks[task_id]

    instances = file_based_config[circuit_file]["instances"]
    seeds = file_based_config[circuit_file]["seeds"]
    max_times = file_based_config[circuit_file]["max_time"]
    if not isinstance(max_times, list):
        max_times = [max_times]

    if DEBUG:
        max_times = [10]
    for max_time in max_times:
        if "optuna" in file_based_config[circuit_file]:
            optlib = "optuna"
        for instance in range(instances):
            with open(f"dataset/{circuit_file}", "rb") as f:
                task = pickle.load(f)

            eq = task[0][instance][0]
            shapes = task[0][instance][1]
            size_dict = task[0][instance][2]
            splitted = eq.split("->")
            inputs = [list(input) for input in splitted[0].split(",")]
            output = list(splitted[1])

            remaining = get_remaining(
                circuit_file,
                instance,
                max_time,
                optlib,
                method,
                seeds,
            )
            print(
                f"For task {instance}, {max_time}, {optlib}, {method} remaining: {remaining}"
            )
            for i in range(remaining):
                repeats = 50000000
                optimizer = ctg.HyperOptimizer(
                    max_repeats=repeats,
                    methods=[method],
                    optlib=optlib,
                    minimize="flops",
                    parallel=True,
                    progbar=True,
                    max_time=max_time,
                )

                start = time.time()
                tree = optimizer.search(inputs, output, size_dict)
                end = time.time()

                search_time = end - start
                print("Search Time:", search_time)

                path, path_info = tree_to_path_info(tree)
                trials = optimizer.get_trials()

                print("Opt cost", path_info.opt_cost)
                print("Trials", len(trials))

                store_path(
                    circuit_file,
                    instance,
                    max_time,
                    optlib,
                    method,
                    search_time,
                    len(trials),
                    path,
                    path_info.opt_cost,
                )
