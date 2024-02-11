import copy
import json

import os
import pickle
import sys
import time
from pathlib import Path

from joblib import Parallel
from opt_einsum.contract import PathInfo

import hybrid_hypercut_greedy as hhg


def serialize_object(obj):
    if isinstance(obj, PathInfo):
        return {
            "path": obj.path,
            "cost": str(obj.opt_cost),
            "largest_intermediate": str(obj.largest_intermediate),
        }

    if isinstance(obj, frozenset):
        return str(set(obj))

    if isinstance(obj, str):
        return obj
    print(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


result_dir = "results/hybrid/"


def get_filename(
    circuit_file,
    max_seconds,
    imbalance,
    weight_function,
    hyper_params,
    instance,
):
    return f"{result_dir}{circuit_file}_{instance}_{max_seconds}_{imbalance}_{weight_function}_{hyper_params}.json"


def get_remaining(
    circuit_file,
    max_seconds,
    imbalance,
    weight_function,
    hyper_params,
    instance,
    target,
):
    filename = get_filename(
        circuit_file, max_seconds, imbalance, weight_function, hyper_params, instance
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
        imbalance,
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


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
circuit_files = [
    # "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_25_mean_conn_3.p", Can be solved optimally, no need to benchmark
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
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_50_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [5, 10, 30, 60],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_75_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [5, 10, 60, 120],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_100_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_125_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_150_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_175_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_200_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "for_paper_TN_d_2_6_dataset_num_eqs_100_num_node_225_mean_conn_3.p": {
        "instances": 100,
        "seeds": 3,
        "max_time": [10, 60, 120, 300],
        "ablation": False,
    },
    "circuit_n53_m10_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
        "ablation": False,
    },
    "circuit_n53_m12_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
        "ablation": False,
    },
    "circuit_n53_m14_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600],
        "ablation": False,
    },
    "circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 360, 600, 1800, 3600, 10800],
        "ablation": False,
    },
    "TN_1688_all.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600, 10800],
        "ablation": False,
    },
    "TN_1688_remaining_steps_1000.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600],
        "ablation": False,
    },
    "TN_1688_remaining_steps_2000.p": {
        "instances": 1,
        "seeds": 10,
        "max_time": [10, 30, 60, 120, 600, 1800, 3600, 10800],
        "ablation": False,
    },
}

imbalances = [[0.05]]

weight_functions_repeats = {
    False: [
        (0, 0, 10),
    ],
    True: [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (5, 0, 0),
        (0, 5, 0),
        (0, 0, 5),
        (5, 5, 10),
    ],
}

hyper_params = {
    True: [
        {
            "one_sided_output": False,
            "num_output_nodes": 0,
            "parts": 2,
        },
        {
            "one_sided_output": True,
            "num_output_nodes": 0,
            "parts": 2,
        },
        {
            "one_sided_output": True,
            "num_output_nodes": 1,
            "parts": 2,
        },
    ],
    False: [
        {
            "one_sided_output": True,
            "num_output_nodes": 1,
            "parts": 2,
        },
    ],
}

tasks = []
for imbalance in imbalances:
    for circuit_file in circuit_files:
        for weight_function in weight_functions_repeats[
            file_based_config[circuit_file]["ablation"]
        ]:
            for hyper_param in hyper_params[
                file_based_config[circuit_file]["ablation"]
            ]:
                tasks.append(
                    (
                        circuit_file,
                        imbalance,
                        weight_function,
                        hyper_param,
                    )
                )

if __name__ == "__main__":
    print(tasks)
    print(len(tasks))
    circuit_file, imbalance, weight_function, hyper_params = tasks[task_id]

    instances = file_based_config[circuit_file]["instances"]
    seeds = file_based_config[circuit_file]["seeds"]
    max_times = file_based_config[circuit_file]["max_time"]
    if not isinstance(max_times, list):
        max_times = [max_times]

    with Parallel(n_jobs=-1) as parallel:
        for max_time in max_times:
            for instance in range(instances):
                with open(f"dataset/{circuit_file}", "rb") as f:
                    task = pickle.load(f)

                eq = task[0][instance][0]
                shapes = task[0][instance][1]
                size_dict = task[0][instance][2]
                splitted = eq.split("->")
                inputs: hhg.Inputs = [list(input) for input in splitted[0].split(",")]
                output = list(splitted[1])

                remaining = get_remaining(
                    circuit_file,
                    max_time,
                    imbalance,
                    weight_function,
                    hyper_params,
                    instance,
                    seeds,
                )
                print(
                    f"For task {instance}, {imbalance}, {weight_function}, {hyper_params} remaining: {remaining}"
                )
                for i in range(remaining):
                    print(
                        "Current task",
                        circuit_file,
                        imbalance,
                        weight_function,
                        hyper_params,
                        instance,
                    )

                    greedy_optimizers = hhg.GreedyOptimizers(
                        hhg.contengrust_greedy(32, parallel),
                        hhg.contengrust_greedy(64, parallel),
                    )

                    cp_unweighted = copy.copy(hhg.unweighted)
                    cp_node_weighted = copy.copy(hhg.node_weight)
                    cp_path_weigthed = copy.copy(hhg.path_weight)

                    cp_unweighted.attempts = weight_function[0]
                    cp_node_weighted.attempts = weight_function[1]
                    cp_path_weigthed.attempts = weight_function[2]
                    weight_functions = [
                        cp_unweighted,
                        cp_node_weighted,
                        cp_path_weigthed,
                    ]

                    start = time.time()
                    path_info, trials, time_result = hhg.repeated_path_finder(
                        inputs,
                        shapes,
                        output,
                        weight_functions=weight_functions,
                        imbalances=list(imbalance),
                        greedy_optimizers=greedy_optimizers,
                        max_time=max_time,
                        **hyper_params,
                    )
                    end = time.time()
                    total_search_time = end - start

                    print("Search Time:", total_search_time)
                    print("Opt cost", path_info.opt_cost)
                    print("Trials", trials)

                    store_path(
                        circuit_file,
                        max_time,
                        instance,
                        imbalance,
                        weight_function,
                        hyper_params,
                        total_search_time,
                        path_info.path,
                        path_info.opt_cost,
                        trials,
                        time_result,
                    )
