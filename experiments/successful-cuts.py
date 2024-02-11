import experiments.hybrid_hypercut_greedy as hhg
from joblib import Parallel
import copy
import time
import pickle
import json

circuit_files = [
    "circuit_n53_m20_s0_e0_pABCDCDAB_simplified_baselines.p",
    "TN_1688_all.p",
]

hyper_params = [
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
]

weight_repeat_tuples = [
    (5, 0, 0),
    (0, 5, 0),
    (0, 0, 5),
]


results = []
with Parallel(n_jobs=-1) as parallel:
    for circuit_file in circuit_files:
        with open(f"dataset/{circuit_file}", "rb") as f:
            task = pickle.load(f)
        eq = task[0][0][0]
        shapes = task[0][0][1]
        size_dict = task[0][0][2]
        splitted = eq.split("->")
        inputs = [list(input) for input in splitted[0].split(",")]
        output = list(splitted[1])
        for hyper_param in hyper_params:
            for weight_repeat_tuple in weight_repeat_tuples:
                for i in range(20):
                    greedy_optimizers = hhg.GreedyOptimizers(
                        hhg.contengrust_greedy(32, parallel),
                        hhg.contengrust_greedy(64, parallel),
                    )

                    cp_unweighted = copy.copy(hhg.unweighted)
                    cp_node_weighted = copy.copy(hhg.node_weight)
                    cp_path_weigthed = copy.copy(hhg.path_weight)

                    cp_unweighted.attempts = weight_repeat_tuple[0]
                    cp_node_weighted.attempts = weight_repeat_tuple[1]
                    cp_path_weigthed.attempts = weight_repeat_tuple[2]
                    weight_functions = [
                        cp_unweighted,
                        cp_node_weighted,
                        cp_path_weigthed,
                    ]

                    start = time.time()
                    path_info, run_infos, successful_cuts = hhg.hybrid_hypercut_greedy(
                        inputs,
                        shapes,
                        output,
                        weight_functions=weight_functions,
                        imbalances=[0.05],
                        greedy_optimizers=greedy_optimizers,
                        **hyper_param,
                    )
                    end = time.time()
                    total_search_time = end - start
                    print(successful_cuts, total_search_time)
                    results.append(
                        {
                            "circuit": circuit_file,
                            "hyper_param": hyper_param,
                            "successful_cuts": successful_cuts,
                            "total_search_time": total_search_time,
                            "weight_function": weight_repeat_tuple,
                            "opt_cost": float(path_info.opt_cost),
                        }
                    )

# Write results to json
with open("successful_cuts.json", "w") as f:
    json.dump(results, f)
