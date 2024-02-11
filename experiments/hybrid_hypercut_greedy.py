import copy
import math
import random
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import total_ordering
from os.path import join
from queue import PriorityQueue
from typing import Callable, Hashable, Literal, Optional, Sequence, Union

import cotengra as ctg
from cotengra.hyperoptimizers.hyper import register_hyper_function
import cotengrust as ctr
import kahypar as kahypar
import opt_einsum as oe
from joblib import Parallel, delayed

# import sesum.sr as sr
from opt_einsum.contract import PathInfo
from opt_einsum.paths import ssa_to_linear
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style

console = Console()

Inputs = list[list[Hashable]]
Shape = tuple[int, ...]
Shapes = list[Shape]
Output = list[Hashable]
SizeDict = dict[Hashable, int]
Histogram = dict[Hashable, int]
Path = list[tuple[int, ...]]
NumOutputNodes = Literal[0, 1]
GreedyOptimizer = Callable[["TensorNetwork"], Path]

debug = False


@dataclass
class BasicInputNode:
    indices: list[Hashable]
    shape: Shape

    def get_size(self) -> int:
        return 1


@dataclass
class OriginalInputNode(BasicInputNode):
    id: int

    def get_id(self):
        return str(self.id)

    def __repr__(self) -> str:
        return f"Original Input({self.indices}, {self.shape})"


@dataclass
class SubNetworkInputNode(BasicInputNode):
    sub_network: "SubTensorNetwork"

    def get_id(self):
        return f"sn-{self.sub_network.name}"

    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices}, {self.sub_network.get_output_shape()})"


InputNode = Union[OriginalInputNode, SubNetworkInputNode]
InputNodes = list[InputNode]


@dataclass
class WeightedBasicInputNode(BasicInputNode):
    weight: int


@dataclass
class WeightedOriginalInputNode(WeightedBasicInputNode, OriginalInputNode):
    def __repr__(self) -> str:
        return f"W Original Input({self.indices}, {self.shape}) weight: {self.weight}"


@dataclass
class WeightedSubNetworkInputNode(WeightedBasicInputNode, SubNetworkInputNode):
    def __repr__(self) -> str:
        return f"Sub network Input({self.sub_network.output_indices}, {self.sub_network.get_output_shape()}) weight: {self.weight}"


WeightedInputNode = Union[WeightedOriginalInputNode, WeightedSubNetworkInputNode]
WeightedInputNodes = list[WeightedInputNode]


def safe_log2(x):
    if x < 1:
        return 0
    return math.log2(x)


def safe_log10(x):
    if x < 1:
        return 0
    return math.log10(x)


def set_weight(node: InputNode, weight):
    if isinstance(node, OriginalInputNode):
        return WeightedOriginalInputNode(
            node.indices,
            node.shape,
            node.id,
            weight,
        )
    elif isinstance(node, SubNetworkInputNode):
        return WeightedSubNetworkInputNode(
            node.indices,
            node.shape,
            node.sub_network,
            weight,
        )
    else:
        raise Exception("Unknown input node type")


@dataclass
class IntermediateContractNode:
    all_indices: set[Hashable]  # Union all indices of children
    scale: int
    indices: Output
    children: list["ContractTreeNode"]
    size: int
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get_id(self):
        return self.uuid

    def get_size(self):
        return self.size

    def __repr__(self) -> str:
        return f"IntermediateContractNode({self.uuid}), children: {[child.get_id() for child in self.children]}"


ContractTreeNode = Union[
    OriginalInputNode, SubNetworkInputNode, IntermediateContractNode
]

ContractTree = list[
    ContractTreeNode
]  # The first n elements of the list are the inputs, afterwards come the intermediate nodes, the last entry is the root


def get_cost_from_path(
    path: Path,
    inputs,
    size_dict: SizeDict,
    histogram: Histogram,
):
    total_cost = 0.0
    inputs = copy.copy(inputs)
    histogram = copy.copy(histogram)
    for pair in path:
        if len(pair) == 1:
            all_indices = set(inputs[pair[0]])

            remove = set()
            for index in inputs[pair[0]]:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += 1
            intermediate = all_indices - remove

            for index in intermediate:
                histogram[index] += 1

            inputs.append(intermediate)
        if len(pair) == 2:
            all_indices = set(inputs[pair[0]]).union(set(inputs[pair[1]]))
            cost = 1.0
            remove = set()
            for index in all_indices:
                cost = cost * size_dict[index]

            for index in inputs[pair[0]]:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            for index in inputs[pair[1]]:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += cost
            intermediate = all_indices - remove

            for index in intermediate:
                histogram[index] += 1

            inputs.append(intermediate)
    return 2 * total_cost


def get_contract_tree_and_cost_from_path(
    tn: "TensorNetwork", ssa_path
) -> tuple[ContractTree, int]:
    contract_tree: ContractTree = []
    histogram = defaultdict(lambda: 0)
    for input in tn.get_all_input_nodes():
        contract_tree.append(input)
        for edge in input.indices:
            histogram[edge] += 1

    for index in tn.output_indices:
        histogram[index] += 1

    # If there is only one input thats the whole tree
    if len(contract_tree) == 1:
        return contract_tree, 1
    total_cost = 0

    for pair in ssa_path:
        if len(pair) == 1:
            left_node: ContractTreeNode = contract_tree[pair[0]]
            all_indices = set(left_node.indices)
            cost = 1
            remove = set()
            for index in left_node.indices:
                cost = cost * tn.size_dict[index]
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += cost
            intermediate = all_indices - remove
            for index in intermediate:
                histogram[index] += 1
            contract_tree.append(
                IntermediateContractNode(
                    all_indices,
                    int(safe_log2(cost)),
                    list(intermediate),
                    [contract_tree[pair[0]]],
                    contract_tree[pair[0]].get_size(),
                )
            )
        if len(pair) == 2:
            left_node: ContractTreeNode = contract_tree[pair[0]]
            right_node: ContractTreeNode = contract_tree[pair[1]]
            all_indices = set(left_node.indices).union(right_node.indices)
            cost = 1
            remove = set()
            for index in all_indices:
                cost = cost * tn.size_dict[index]

            for index in left_node.indices:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            for index in right_node.indices:
                histogram[index] -= 1
                if histogram[index] == 0:
                    remove.add(index)
            total_cost += cost
            intermediate = all_indices - remove

            for index in intermediate:
                histogram[index] += 1

            contract_tree.append(
                IntermediateContractNode(
                    all_indices,
                    int(safe_log2(cost)),
                    list(intermediate),
                    [contract_tree[pair[0]], contract_tree[pair[1]]],
                    contract_tree[pair[0]].get_size()
                    + contract_tree[pair[1]].get_size(),
                )
            )
    total_cost = 2 * total_cost

    return contract_tree, total_cost


def random_cotengrust_greedy(
    inputs: Inputs,
    output: Output,
    size_dict: SizeDict,
    histogram: Histogram,
):
    best_cost = math.inf
    best_path = None
    for i in range(8):
        temp = 2 ** random.uniform(math.log2(0.001), math.log2(1))
        path = ctr.optimize_greedy(  # type: ignore
            inputs,
            output,
            size_dict,
            costmod=random.uniform(0, 50),
            temperature=temp,
            simplify=True,
            use_ssa=True,
        )
        cost = get_cost_from_path(path, inputs, size_dict, histogram)
        if cost < best_cost:
            best_cost = cost
            best_path = path
    return best_path, best_cost


def build_initial_tn(inputs: Inputs, output: Output, shapes: Shapes):
    indices = frozenset.union(*[frozenset(input) for input in inputs])
    size_dict = ctg.utils.shapes_inputs_to_size_dict(shapes, inputs)

    input_nodes: InputNodes = [
        OriginalInputNode(in_sh[0], in_sh[1], id)
        for id, in_sh in enumerate(zip(inputs, shapes))
    ]

    return SubTensorNetwork(
        "tn", 0, None, input_nodes, indices, size_dict, frozenset(), output
    )


@dataclass
class SubTensorNetwork:
    name: str
    key: int
    parent_name: str | None
    inputs: InputNodes
    indices: frozenset[Hashable]
    size_dict: SizeDict
    cut_indices: frozenset[Hashable]
    output_indices: Output

    def get_all_input_nodes(self) -> InputNodes:
        return self.inputs

    def get_all_networks(self):
        return [self]

    def get_output_shape(self):
        return tuple([self.size_dict[index] for index in self.output_indices])

    def find_path(self, greedy_optimizer: GreedyOptimizer):
        """
        Finds the path for the sub-tensor network.

        Returns:
            SubTensorNetworkWithContractTree: The sub-tensor network with the computed contract tree and its cost.
        """
        path = greedy_optimizer(self)

        contract_tree, cost = get_contract_tree_and_cost_from_path(self, path)

        tn_with_tree = SubTensorNetworkWithContractTree(
            name=self.name,
            key=self.key,
            parent_name=self.parent_name,
            inputs=self.inputs,
            indices=self.indices,
            size_dict=self.size_dict,
            cut_indices=self.cut_indices,
            output_indices=self.output_indices,
            cost=cost,
            contract_tree=contract_tree,
        )
        return tn_with_tree


@total_ordering
@dataclass
class SubTensorNetworkWithContractTree(SubTensorNetwork):
    cost: int
    contract_tree: ContractTree

    def get_total_cost(self):
        return self.cost

    def get_contract_tree(self):
        return self.contract_tree

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return self.cost == other.cost

    def __lt__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        # Yes this might seem weird, but we want the one with the highest cost to be the first in the priority queue
        return self.cost > other.cost

    def print_stats(self):
        biggest_input = (
            max([len(input.indices) for input in self.inputs])
            if len(self.inputs) > 0
            else 0
        )
        print_md(
            f"Network {self.name}: {len(self.inputs)} inputs, {len(self.indices)} indices, {len(self.output_indices)} output indices, {self.cost:.4e} log10 {safe_log10(self.cost)} cost, biggest input {biggest_input}"
        )

    def find_path(self, greedy_optimizer: GreedyOptimizer):
        """
        Since the path is already computed, this function just returns the current object.

        Returns:
            The current object.
        """
        return self

    def refine_tree(self, tree: ContractTree, cost: int):
        self.contract_tree = tree
        self.cost = cost


@dataclass
class SuperTensorNetwork(SubTensorNetwork):
    parent_name: str
    sub_networks: Sequence[SubTensorNetwork]

    def get_all_input_nodes(self) -> InputNodes:
        sub_input_nodes = [
            SubNetworkInputNode(
                sub_network.output_indices, sub_network.get_output_shape(), sub_network
            )
            for sub_network in self.sub_networks
        ]

        return sub_input_nodes + self.inputs

    def get_all_networks(self):
        return [self] + [sub_network for sub_network in self.sub_networks]

    def find_path(self, greedy_optimizer: GreedyOptimizer):
        sub_networks_with_path: list[SubTensorNetworkWithContractTree] = []
        for sub_network in self.sub_networks:
            sub_networks_with_path.append(sub_network.find_path(greedy_optimizer))

        self.sub_networks = sub_networks_with_path

        path = greedy_optimizer(self)
        contract_tree, cost = get_contract_tree_and_cost_from_path(self, path)

        tn_with_tree = SuperTensorNetworkWithTree(
            self.name,
            self.key,
            self.parent_name,
            self.inputs,
            self.indices,
            self.size_dict,
            self.cut_indices,
            self.output_indices,
            cost,
            contract_tree,
            sub_networks_with_path,
        )

        return tn_with_tree


@dataclass
class SuperTensorNetworkWithTree(
    SuperTensorNetwork,
    SubTensorNetworkWithContractTree,
):
    sub_networks: list[SubTensorNetworkWithContractTree]

    def print_stats(self):
        print_md(f"#### Block stats, for sub blocks of {self.parent_name}")

        print_md("Super Problem:")
        super().print_stats()
        biggest_sub_output = max(
            [len(sub_network.output_indices) for sub_network in self.sub_networks]
        )
        print_md(f"Biggest sub output {biggest_sub_output}")
        print_md("Sub Problems:")
        for sub_network in self.sub_networks:
            sub_network.print_stats()

    def find_path(self, greedy_optimizer: GreedyOptimizer):
        """
        Since the path is already computed, this function just returns the current object.

        Returns:
            The current object.
        """
        return self

    def get_total_cost(self):
        return sum([sub_network.cost for sub_network in self.sub_networks]) + self.cost

    def get_parent_tree(self):
        super_tree = self.get_contract_tree()

        parent_tree = []
        sub_tree_root: dict[str, ContractTreeNode] = {}
        for node in super_tree:
            if (
                isinstance(node, SubNetworkInputNode)
                and node.sub_network.parent_name == self.name
            ):
                assert isinstance(
                    node.sub_network, SubTensorNetworkWithContractTree
                ), "The subnetworks should have a contract tree, when calling get_parent_tree"
                sub_tree = None
                for sn in self.sub_networks:
                    if sn.name == node.sub_network.name:
                        sub_tree = sn.contract_tree
                assert (
                    sub_tree is not None
                ), f"Sub tree {node.sub_network.name} not found in {self.name}"
                for sub_node in sub_tree:
                    parent_tree.append(sub_node)

                sub_tree_root[node.sub_network.name] = sub_tree[-1]
            elif isinstance(node, IntermediateContractNode):
                for key, child in enumerate(node.children):
                    if (
                        isinstance(child, SubNetworkInputNode)
                        and child.sub_network.parent_name == self.name
                    ):
                        node.children[key] = sub_tree_root[child.sub_network.name]

                parent_tree.append(node)
            else:
                parent_tree.append(node)

        return parent_tree

    def update_tree(self, name, tree: ContractTree):
        if name == self.name:
            self.contract_tree = tree
        else:
            for sub_network in self.sub_networks:
                if sub_network.name == name:
                    sub_network.contract_tree = tree
                    return
            raise Exception(f"name {name} not found in {self.name}")


TensorNetwork = Union[SubTensorNetwork, SuperTensorNetwork]
TensorNetworkWithTree = Union[
    SubTensorNetworkWithContractTree, SuperTensorNetworkWithTree
]


def print_md(
    *markdowns,
    style: Optional[Union[str, Style]] = None,
):
    if debug:
        for markdown in markdowns:
            console.print(Markdown(f"{markdown}"), style=style)


def get_path_info_from_path(path, eq, shapes):
    path, path_info = oe.contract_path(eq, *shapes, shapes=True, optimize=path)

    # check that path_info is from PathInfo class
    assert isinstance(path_info, PathInfo)

    return path_info


# Partially based on code from cotengra, for license see: https://github.com/jcmgray/cotengra/blob/main/LICENSE.md
def partition_tn(
    input_nodes: WeightedInputNodes,
    output_node: WeightedBasicInputNode,
    size_dict,
    parts=2,
    imbalance=0.1,
    seed=None,
    profile=None,
    mode="recursive",
    objective="cut",
    num_output_nodes: NumOutputNodes = 1,
    fixed_inputs_by_block: dict[
        int, list[int]
    ] = {},  # Dict inputs that should be fixed to the block in the key
):
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    output = output_node.indices
    one_output_node = num_output_nodes > 0 and len(output) > 0
    output_weight = output_node.weight

    hyper_edges = defaultdict(lambda: [])

    # Filter out nodes, that have only output indices
    all_nodes = []
    all_node_weights = []
    for input_node in input_nodes:
        all_nodes.append(input_node.indices)
        all_node_weights.append(input_node.weight)

    number_of_inputs = len(all_nodes)
    if parts >= number_of_inputs:
        return list(range(number_of_inputs))

    if one_output_node:
        all_nodes.append(list(output))
        all_node_weights.append(output_weight)

    for node_index, input in enumerate(all_nodes):
        for index in input:
            hyper_edges[index].append(node_index)

    # Filter out open edges
    hyper_edge_list = [e for e in hyper_edges.values() if len(e) > 1]
    hyper_edge_keys = [key for key, e in hyper_edges.items() if len(e) > 1]
    if len(hyper_edge_list) == 0:
        num_of_all_nodes = len(all_nodes)
        return [
            i // (math.ceil(num_of_all_nodes / (parts)))
            for i in range(num_of_all_nodes)
        ]
    index_vector = []
    edge_vector = []

    for e in hyper_edge_list:
        index_vector.append(len(edge_vector))
        edge_vector.extend(e)

    index_vector.append(len(edge_vector))

    edge_weights = [
        int(max(safe_log2((size_dict[edge])), 1)) for edge in hyper_edge_keys
    ]

    hypergraph_kwargs = {
        "num_nodes": len(all_nodes),
        "num_edges": len(hyper_edge_list),
        "index_vector": index_vector,
        "edge_vector": edge_vector,
        "k": parts,
        "edge_weights": edge_weights,
        "node_weights": all_node_weights,
    }

    hypergraph = kahypar.Hypergraph(**hypergraph_kwargs)

    for block_id, fixed_inputs in fixed_inputs_by_block.items():
        for fixed_input in fixed_inputs:
            hypergraph.fixNodeToBlock(fixed_input, block_id)

    if profile is None:
        profile_mode = {"direct": "k", "recursive": "r"}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"

    context = kahypar.Context()
    context.loadINIconfiguration(
        join(ctg.path_kahypar.get_kahypar_profile_dir(), profile)
    )
    context.setK(parts)
    context.setSeed(seed)
    context.suppressOutput(True)
    context.setEpsilon(imbalance)

    kahypar.partition(hypergraph, context)
    print_md(f"Kahypar, cut value: {kahypar.cut(hypergraph)}")

    return [hypergraph.blockID(i) for i in hypergraph.nodes()]


def get_sub_networks(
    tensor_network_name: str,
    input_nodes: WeightedInputNodes,
    output: WeightedBasicInputNode,
    size_dict: SizeDict,
    imbalance=0.001,
    one_sided_output=True,
    parts=2,
    fixed_inputs_by_block: dict[int, list[int]] = {},
    num_output_nodes: NumOutputNodes = 1,
):
    """
    Splits a given tensor network into sub-networks based on certain criteria.

    Args:
        tensor_network (TensorNetwork): The input tensor network.
        imbalance (float, optional): The imbalance threshold for partitioning the network. Defaults to 0.001.
        input_weights (list[float], optional): The weights assigned to each input. Defaults to None.
        one_sided_output (bool, optional): Whether to consider only one-sided output. Defaults to True.
        parts (int, optional): The number of parts to split the network into. Defaults to 2.
        fixed_inputs_by_block (dict[int, list[int]], optional): Dictionary mapping block IDs to a list of fixed input IDs for each block. Defaults to {}.

    Returns:
        tuple: A tuple containing the following:
            - sub_problems (list[SubTensorNetwork]): List of sub-networks generated from the split.
            - cut_indices (frozenset): Set of indices representing the cut between sub-networks.
            - output_block_id (int): The block ID of the output sub-network.
            - input_remap (dict[int, dict[int, int]]): Dictionary mapping block IDs to a dictionary mapping original input IDs to remapped input IDs.
    """

    if one_sided_output == False:
        assert (
            num_output_nodes != 1
        ), "If the output is not one sided you cannot use a single output node"

    num_input_nodes = len(input_nodes)
    assert (
        num_input_nodes > 2
    ), f"Not enough input nodes to split, pass at least two input nodes, {input_nodes}"

    assert parts >= 2, "Parts must be at least 2"

    block_ids = partition_tn(
        input_nodes,
        output,
        size_dict=size_dict,
        imbalance=imbalance,
        parts=parts,
        num_output_nodes=num_output_nodes,
        fixed_inputs_by_block=fixed_inputs_by_block,
    )

    input_block_ids = block_ids[:num_input_nodes]
    if min(input_block_ids) == max(input_block_ids):
        print_md(f"Only one block of input_nodes found, just split by modulo")
        block_ids = [i % parts for i in range(num_input_nodes + num_output_nodes)]

    # Group inputs by block id
    block_inputs: defaultdict[int, InputNodes] = defaultdict(lambda: [])
    for block_id, input_node in zip(block_ids, input_nodes):
        block_inputs[block_id].append(input_node)

    max_key = max(block_inputs.keys())
    next_key = max_key + 1
    new_block_inputs: defaultdict[int, InputNodes] = defaultdict(lambda: [])
    for key, inputs in block_inputs.items():
        sub_graphs = ctr.find_subgraphs(
            [list(input.indices) for input in inputs], [], size_dict
        )
        if len(sub_graphs) > 1:
            keys = [key] + list(range(next_key, next_key + len(sub_graphs) - 1))
            next_key = next_key + len(sub_graphs) - 1
            for graph, key in zip(sub_graphs, keys):
                sub_inputs = []
                for input_id in graph:
                    sub_inputs.append(inputs[input_id])
                new_block_inputs[key] = sub_inputs
                next_key += 1
        else:
            new_block_inputs[key] = inputs

    block_inputs = new_block_inputs

    block_indices = {
        key: frozenset(set.union(*[set(val.indices) for val in value]))
        for key, value in block_inputs.items()
    }

    assert (
        len(block_inputs.keys()) > 1
    ), "Only one block found, should have been catached above"

    cut_indices = set()
    for block_id1 in block_inputs.keys():
        for block_id2 in block_inputs.keys():
            if block_id1 != block_id2:
                cut_indices = cut_indices.union(
                    block_indices[block_id1].intersection(block_indices[block_id2])
                )

    cut_indices = frozenset(cut_indices.union(output.indices))
    output_indices: dict[int, Output] = {}

    for block_id in block_inputs.keys():
        output_indices[block_id] = list(
            cut_indices.intersection(block_indices[block_id])
        )

    output_block_id = None
    if len(output.indices) > 0 and one_sided_output:
        output_block_id = block_ids[-1]

    sub_problems = {
        key: SubTensorNetwork(
            f"{tensor_network_name}.{key}",
            key,
            None,
            block_inputs[key],
            block_indices[key],
            size_dict,
            cut_indices,
            output_indices[key],
        )
        for key in sorted(block_inputs.keys())
    }

    return sub_problems, cut_indices, output_block_id


def build_super_network(
    super_network: TensorNetwork,
    parent_name: str,
    cut_indices: frozenset[Hashable],
    output: Output,
    sub_networks: list[SubTensorNetwork],
):
    # Set parent for sub_networks
    for sub_network in sub_networks:
        sub_network.parent_name = super_network.name

    super_network = SuperTensorNetwork(
        name=super_network.name,
        key=super_network.key,
        parent_name=parent_name,
        inputs=super_network.inputs,
        indices=super_network.indices,
        size_dict=super_network.size_dict,
        cut_indices=cut_indices,
        output_indices=output,
        sub_networks=sub_networks,
    )
    return super_network


def find_super_network(
    tensor_network: TensorNetwork,
    one_sided_output: bool,
    sub_networks: dict[int, SubTensorNetwork],
    cut_indices: frozenset[Hashable],
    output_block_id: Optional[int],
    greedy_optimizer: GreedyOptimizer,
) -> SuperTensorNetworkWithTree:
    output = tensor_network.output_indices

    if one_sided_output:
        if output_block_id == None:
            print("Test where output should be placed")
            keys = range(len(sub_networks.keys()))
            best_costs = math.inf
            best_super_network = None
            for key in keys:
                new_sub_networks = [
                    copy.deepcopy(sub_network) for sub_network in sub_networks.values()
                ]
                super_sub_network = new_sub_networks.pop(key)

                super_network = build_super_network(
                    super_sub_network,
                    tensor_network.name,
                    cut_indices,
                    output,
                    new_sub_networks,
                )

                super_network_with_path = super_network.find_path(greedy_optimizer)
                total_costs = super_network_with_path.get_total_cost()
                print_md(f"Found total costs {total_costs:.6e}")
                if total_costs < best_costs:
                    best_costs = total_costs
                    best_super_network = super_network_with_path
                    print_md(f"New best costs {best_costs:.6e}")
            assert best_super_network != None
            return best_super_network

        else:
            # Remove sub_network with output from sub_networks and take it as super network
            super_sub_network = sub_networks.pop(output_block_id)
            new_sub_networks = [sub_network for sub_network in sub_networks.values()]

            super_network = build_super_network(
                super_sub_network,
                tensor_network.name,
                cut_indices,
                output,
                new_sub_networks,
            )

            return super_network.find_path(greedy_optimizer)

    else:
        new_super_key = len(sub_networks)
        new_name = f"{tensor_network.name}.{new_super_key}"

        new_sub_networks = []
        # Set parent for sub_networks
        for sub_network in sub_networks.values():
            sub_network.parent_name = new_name
            new_sub_networks.append(sub_network)

        # Just take all subproblems and put them in a superproblem
        # Thus the superproblem has no inputs itself, just the outputs of the subproblems as always
        super_network = SuperTensorNetwork(
            name=new_name,
            key=new_super_key,
            parent_name=tensor_network.name,
            inputs=[],
            indices=cut_indices,
            cut_indices=cut_indices,
            output_indices=output,
            size_dict=tensor_network.size_dict,
            sub_networks=new_sub_networks,
        )
        return super_network.find_path(greedy_optimizer)


def get_eq(inputs, output):
    return ",".join(["".join(input) for input in inputs]) + "->" + "".join(output)


@dataclass
class Partition:
    weight_function_name: str
    imbalance: float
    one_sided_output: bool
    num_output_nodes: NumOutputNodes
    network: SuperTensorNetworkWithTree


@dataclass
class SuperRunInfo:
    problem_name: str
    child_names: list[str]
    best_partition: Partition
    old_cost: int
    new_cost: int
    all_partitions: list[Partition] = field(default_factory=list)
    is_super_problem: Literal[True] = True

    def __str__(self):
        return f"SuperRunInfo({self.problem_name}, children: {self.child_names}, old_cost: {self.old_cost:.6e}, new_cost: {self.new_cost:.6e} cost: {self.best_partition.network.cost:.6e}, total_cost: {self.best_partition.network.get_total_cost():.6e})"


AbortionReason = Literal["cost_too_low", "too_few_inputs", "worse_than_parent"]


@dataclass
class SubRunInfo:
    problem_name: str
    improvement: Literal["refined_parent", "none"]
    abortion_reason: AbortionReason
    old_cost: int
    new_cost: int
    all_partitions: list[Partition] = field(default_factory=list)
    is_super_problem: Literal[False] = False

    def __str__(self):
        return f"SubRunInfo({self.problem_name}, {self.improvement}, {self.abortion_reason}, old_cost: {self.old_cost:.6e}, new_cost: {self.new_cost:.6e})"


RunInfo = Union[SuperRunInfo, SubRunInfo]


def solve_with_more_precision(
    network: TensorNetworkWithTree,
    greedy_optimizer: GreedyOptimizer,
    abortion_reason: AbortionReason,
):
    print_md("Solve parent with higher precision")
    refined_path = greedy_optimizer(network)
    refined_tree, refined_cost = get_contract_tree_and_cost_from_path(
        network, refined_path
    )
    if refined_cost < network.cost:
        print_md("Refined parent better than original parent", style="bold green")
        run_info = SubRunInfo(
            problem_name=network.name,
            improvement="refined_parent",
            abortion_reason=abortion_reason,
            old_cost=network.cost,
            new_cost=refined_cost,
        )
        network.refine_tree(refined_tree, refined_cost)
        return run_info
    else:
        print_md("Refined parent worse than original parent", style="bold red")
        run_info = SubRunInfo(
            problem_name=network.name,
            improvement="none",
            abortion_reason=abortion_reason,
            old_cost=network.cost,
            new_cost=network.cost,
        )
        return run_info


@dataclass
class WeightFunction:
    name: str
    get_weighted_inputs: Callable[[TensorNetwork, ContractTree], WeightedInputNodes]
    get_weighted_ouptut: Callable[[TensorNetwork], WeightedBasicInputNode]
    attempts: int = 0
    averaged: bool = False


unweighted = WeightFunction(
    "unweighted",
    lambda tn, _: [set_weight(input, 1) for input in tn.get_all_input_nodes()],
    lambda tn: WeightedBasicInputNode(tn.output_indices, tn.get_output_shape(), 1),
)


def get_node_weight_by_log_size(tn: TensorNetwork, input: Output):
    return max(sum([int(safe_log2((tn.size_dict[edge]))) for edge in input]), 1)


node_weight = WeightFunction(
    "node_weight",
    lambda tn, _: [
        set_weight(
            input,
            get_node_weight_by_log_size(tn, input.indices),
        )
        for input in tn.get_all_input_nodes()
    ],
    lambda tn: WeightedBasicInputNode(
        tn.output_indices,
        tn.get_output_shape(),
        get_node_weight_by_log_size(tn, tn.output_indices),
    ),
    0,
)


def get_remapped_id(id, input_remap: Optional[dict[str, int]]):
    return input_remap[id] if input_remap != None and id in input_remap else id


def contract_tree_to_path(tree: ContractTree, remap: Optional[dict[str, int]] = None):
    root = tree[-1]

    if isinstance(root, BasicInputNode):
        assert (
            len(tree) == 1
        ), "Tree should only contain one node, if root is basic input"
        return [(int(get_remapped_id(root.get_id(), remap)),)]

    path = []

    counter = (len(tree) + 1) // 2
    uuid_to_ssa_id = {}
    for node in tree:
        if isinstance(node, IntermediateContractNode):
            uuid_to_ssa_id[node.get_id()] = counter
            counter += 1
            pair = []
            if isinstance(node.children[0], BasicInputNode):
                pair.append(int(get_remapped_id(node.children[0].get_id(), remap)))
            else:
                pair.append(uuid_to_ssa_id[node.children[0].get_id()])

            if len(node.children) > 1:
                if isinstance(node.children[1], BasicInputNode):
                    pair.append(int(get_remapped_id(node.children[1].get_id(), remap)))
                else:
                    pair.append(uuid_to_ssa_id[node.children[1].get_id()])
            path.append(tuple(pair))
    return path


def weight_from_tree(tn: TensorNetwork, tree: ContractTree):
    max_weight = defaultdict(lambda: 1)

    all_children = defaultdict(lambda: [])
    for input in tn.get_all_input_nodes():
        max_weight[input.get_id()] = max(
            max_weight[input.get_id()],
            get_node_weight_by_log_size(tn, input.indices),
        )
        all_children[input.get_id()] = [input]

    for node in tree:
        if isinstance(node, IntermediateContractNode):
            all_children[node.get_id()] = all_children[node.children[0].get_id()] + (
                all_children[node.children[1].get_id()]
                if len(node.children) > 1
                else []
            )
            for child in all_children[node.get_id()]:
                weight = node.scale * (
                    len(node.all_indices.intersection(child.indices))
                )
                max_weight[child.get_id()] = max(max_weight[child.get_id()], weight)

    weighted_inputs = [
        set_weight(input, int(max_weight[input.get_id()]))
        for input in tn.get_all_input_nodes()
    ]
    return weighted_inputs


path_weight = WeightFunction(
    "path_weight",
    weight_from_tree,
    lambda tn: WeightedBasicInputNode(
        tn.output_indices,
        tn.get_output_shape(),
        len(tn.output_indices) * get_node_weight_by_log_size(tn, tn.output_indices),
    ),
    5,
    True,
)


@dataclass
class GreedyOptimizers:
    quick: GreedyOptimizer
    long: GreedyOptimizer


def oe_greedy(tn: TensorNetwork):
    eq = get_eq(
        [input.indices for input in tn.get_all_input_nodes()], tn.output_indices
    )
    shapes = [input.shape for input in tn.get_all_input_nodes()]
    path, path_info = oe.contract_path(eq, *shapes, shapes=True, optimize="greedy")
    assert isinstance(path_info, PathInfo)
    return path_info


def contengrust_greedy(repeats: int = 32, parallel=None):
    assert repeats >= 8, "Repeats must be at least 8"

    def greedy_optimizer(tn: TensorNetwork) -> Path:
        inputs = [input.indices for input in tn.get_all_input_nodes()]
        eq = get_eq(inputs, tn.output_indices)
        output = tn.output_indices
        size_dict = tn.size_dict
        shapes = [input.shape for input in tn.get_all_input_nodes()]

        if len(inputs) <= 15:
            print("Find optimal path")
            optimal_path = ctr.optimize_optimal(
                inputs,
                output,
                size_dict,
                minimize="flops",
                search_outer=False,
                use_ssa=True,
            )
            return optimal_path

        histogram = defaultdict(lambda: 0)

        for index in inputs:
            for edge in index:
                histogram[edge] += 1

        for index in output:
            histogram[index] += 1

        results = []

        if parallel is None:
            results = Parallel(n_jobs=-1)(
                delayed(random_cotengrust_greedy)(inputs, output, size_dict, histogram)
                for _ in range(repeats)
            )
        else:
            results = parallel(
                delayed(random_cotengrust_greedy)(inputs, output, size_dict, histogram)
                for _ in range(repeats)
            )

        best_cost = math.inf
        best_path = None

        for path, cost in results:  # type: ignore
            if cost < best_cost:
                best_cost = cost
                best_path = path

        return best_path  # type: ignore
        # path_info = get_path_info_from_path(ctr.ssa_to_linear(best_path), eq, shapes)  # type: ignore
        # return path_info

    return greedy_optimizer


fast_repeats = 32
slow_repeat = 64

default_greedys = GreedyOptimizers(
    contengrust_greedy(fast_repeats),
    contengrust_greedy(slow_repeat),
)


def hybrid_hypercut_greedy(
    inputs: Inputs,
    shapes: Shapes,
    output: Output,
    imbalances: list[float] = [0.05],
    one_sided_output=True,
    num_output_nodes: NumOutputNodes = 1,
    parts: int = 2,
    weight_functions: list[WeightFunction] = [unweighted, node_weight, path_weight],
    greedy_optimizers: GreedyOptimizers = default_greedys,
    split_cost_threshold=10 ** (-5),
    cutoff=15,
) -> tuple[PathInfo, dict[str, RunInfo], int]:
    # Problem setup, transform arguments to tanser network with path
    indices = frozenset.union(*[frozenset(input) for input in inputs])
    size_dict = ctg.utils.shapes_inputs_to_size_dict(shapes, inputs)

    input_nodes: InputNodes = [
        OriginalInputNode(in_sh[0], in_sh[1], id)
        for id, in_sh in enumerate(zip(inputs, shapes))
    ]

    tensor_network_without_path = SubTensorNetwork(
        "tn", 0, None, input_nodes, indices, size_dict, frozenset(), output
    )

    tensor_network = tensor_network_without_path.find_path(greedy_optimizers.quick)

    root_name = tensor_network.name

    # Initialize queues
    to_partition: PriorityQueue[TensorNetworkWithTree] = PriorityQueue()
    to_partition.put(tensor_network)

    # Initialize dictoionary for finalized partitions
    partitioned: dict[str, TensorNetworkWithTree] = {}
    run_infos: dict[str, RunInfo] = {}

    # Track current cost
    current_cost = tensor_network.cost
    print(f"First greedy cost {current_cost:.6e}")

    successfull_cuts = 0

    def finalize(network: TensorNetworkWithTree, old_cost, run_info: RunInfo):
        nonlocal current_cost
        current_cost = current_cost + network.cost - old_cost
        partitioned[network.name] = network
        run_infos[network.name] = run_info

    while not to_partition.empty():
        next_network = to_partition.get()
        if len(next_network.inputs) <= cutoff:
            print_md(f"==> Few inputs, skip splitting {next_network.name}")
            run_info = SubRunInfo(
                problem_name=next_network.name,
                improvement="none",
                abortion_reason="too_few_inputs",
                old_cost=next_network.cost,
                new_cost=next_network.cost,
            )
            finalize(next_network, next_network.cost, run_info)
            continue

        if next_network.cost / current_cost < split_cost_threshold:
            print_md(f"==> Cost too low, stop splitting {next_network.name}")
            run_info = SubRunInfo(
                problem_name=next_network.name,
                improvement="none",
                abortion_reason="cost_too_low",
                old_cost=next_network.cost,
                new_cost=next_network.cost,
            )
            old_cost = next_network.cost
            finalize(next_network, old_cost, run_info)
            while not to_partition.empty():
                to_finalize = to_partition.get()
                print_md(f"==> Cost too low, stop splitting {to_finalize.name}")
                run_info = SubRunInfo(
                    problem_name=to_finalize.name,
                    improvement="none",
                    abortion_reason="cost_too_low",
                    old_cost=to_finalize.cost,
                    new_cost=to_finalize.cost,
                )
                old_cost = to_finalize.cost
                finalize(to_finalize, old_cost, run_info)
            break

        print_md(
            f"## Run for {next_network.name} {next_network.cost:.4e}, log10 = {safe_log10(next_network.cost):.4f}"
        )

        best_partition: Optional[Partition] = None
        all_partitions: list[Partition] = []
        tested_cuts = []
        for imbalance in imbalances:
            for weight_function in weight_functions:
                weighted_input_nodes = weight_function.get_weighted_inputs(
                    next_network, next_network.get_contract_tree()
                )
                assert (
                    len(weighted_input_nodes) > 2
                ), f"Not enough input nodes, {weighted_input_nodes}"
                weighted_output_node = weight_function.get_weighted_ouptut(next_network)
                input_weight_sum = weighted_input_nodes
                for attempt in range(weight_function.attempts):
                    print_md(
                        f'##### Attempt {attempt+1} for weight function "{weight_function.name}"'
                    )
                    (sub_networks, cut_indices, output_block_id) = get_sub_networks(
                        next_network.name,
                        weighted_input_nodes,
                        weighted_output_node,
                        size_dict,
                        imbalance=imbalance,
                        one_sided_output=one_sided_output,
                        num_output_nodes=num_output_nodes,
                        parts=parts,
                    )
                    if cut_indices in tested_cuts:
                        print_md("Same cut was found before, continue")
                        continue

                    tested_cuts.append(cut_indices)

                    super_network = find_super_network(
                        next_network,
                        one_sided_output,
                        sub_networks,
                        cut_indices,
                        output_block_id,
                        greedy_optimizer=greedy_optimizers.quick,
                    )
                    super_network.print_stats()

                    partition = Partition(
                        weight_function.name,
                        imbalance,
                        one_sided_output,
                        num_output_nodes,
                        copy.deepcopy(super_network),
                    )

                    all_partitions.append(partition)

                    if (best_partition == None) or (
                        super_network.get_total_cost()
                        < best_partition.network.get_total_cost()
                    ):
                        print_md(
                            f"===============> Found new best partition: {super_network.get_total_cost():.4e} log10({safe_log10(super_network.get_total_cost())})",
                            style="bold green",
                        )
                        best_partition = partition

                    if (
                        weight_function.averaged
                        and attempt < weight_function.attempts - 1
                    ):
                        new_input_weights = weight_function.get_weighted_inputs(
                            next_network,
                            super_network.get_parent_tree(),
                        )
                        assert new_input_weights != None
                        assert input_weight_sum != None
                        input_weight_sum = [
                            set_weight(a, a.weight + b.weight)
                            for a, b in zip(input_weight_sum, new_input_weights)
                        ]
                        # Divide by number of attempts
                        weighted_input_nodes = [
                            set_weight(node, max(int(node.weight / (attempt + 2)), 1))
                            for node in input_weight_sum
                        ]

        assert (
            best_partition != None
        ), "No best partition found, no weight functions or imbalances passed?"

        best_partition.network.print_stats()

        if best_partition.network.get_total_cost() >= next_network.cost:
            print_md(
                "==> Subproblem costs higher than parent, stop splitting",
                style="bold red",
            )
            old_cost = next_network.cost
            run_info = solve_with_more_precision(
                next_network,
                greedy_optimizers.long,
                "worse_than_parent",
            )
            run_info.all_partitions = all_partitions
            finalize(next_network, old_cost, run_info)
            continue

        to_partition.put(best_partition.network)
        successfull_cuts = successfull_cuts + 1

        run_info = SuperRunInfo(
            problem_name=best_partition.network.name,
            child_names=[
                sub_network.name for sub_network in best_partition.network.sub_networks
            ],
            best_partition=best_partition,
            old_cost=next_network.cost,
            new_cost=best_partition.network.get_total_cost(),
            all_partitions=all_partitions,
        )

        run_infos[best_partition.network.parent_name] = run_info
        partitioned[best_partition.network.parent_name] = best_partition.network

        for sub_network in best_partition.network.sub_networks:
            to_partition.put(sub_network)

        current_cost = (
            current_cost - next_network.cost + best_partition.network.get_total_cost()
        )
        print(f"## Finished partitioning: {next_network.name}")
        print(
            f"Current cost {current_cost:.4e}, log10 = {safe_log10(current_cost):.4f}"
        )
    print_md("## Finished partitioning, start merging", style="bold green")
    print(f"Current cost {current_cost:.4e}, log10 = {safe_log10(current_cost):.4f}")

    def merge(network_name) -> ContractTree:
        partitioned_network = partitioned[network_name]
        print_md(f"Start Merge {partitioned_network.name}")
        # Check if we reached a leave
        if partitioned_network.name == network_name or not (
            isinstance(partitioned_network, SuperTensorNetworkWithTree)
        ):
            print_md(f"==> {partitioned_network.name} is leave, stop merging")
            return partitioned_network.contract_tree
        for sub_network in partitioned_network.get_all_networks():
            print_md(f"==> Call merge for sub_network {sub_network.name}")
            merged_sub_tree = merge(sub_network.name)
            partitioned_network.update_tree(sub_network.name, merged_sub_tree)
        print_md(f"Finished Merge {partitioned_network.name}")

        parent_tree = partitioned_network.get_parent_tree()

        return parent_tree

    parent_tree = merge(root_name)
    path = contract_tree_to_path(parent_tree)
    path_info = get_path_info_from_path(
        ssa_to_linear(path),
        get_eq(inputs, output),
        shapes,
    )

    return path_info, run_infos, successfull_cuts


def repeated_path_finder(
    inputs: Inputs,
    shapes: Shapes,
    output: Output,
    max_repeats: Optional[int] = None,
    max_time: float = math.inf,
    imbalances: list[float] = [0.05],
    weight_functions=[unweighted, node_weight, path_weight],
    one_sided_output=True,
    num_output_nodes: NumOutputNodes = 1,
    parts: int = 2,
    greedy_optimizers: GreedyOptimizers = default_greedys,
    split_cost_threshold=10 ** (-5),
    cutoff=15,
):
    assert (max_repeats != None and max_repeats > 0) or (
        max_time != math.inf
    ), "Either repeats or max_time must be set"

    total_search_time = 0
    last_search_time = 0
    best_path_info = None
    trials = 0
    time_result = []
    while total_search_time + last_search_time < max_time and (
        max_repeats == None or trials < max_repeats
    ):
        start = time.time()
        path_info, run_infos, successful_cuts = hybrid_hypercut_greedy(
            inputs,
            shapes,
            output,
            weight_functions=weight_functions,
            imbalances=imbalances,
            greedy_optimizers=greedy_optimizers,
            one_sided_output=one_sided_output,
            num_output_nodes=num_output_nodes,
            parts=parts,
            split_cost_threshold=split_cost_threshold,
            cutoff=cutoff,
        )
        end = time.time()
        last_search_time = end - start
        total_search_time = total_search_time + last_search_time
        trials = trials + 1

        if best_path_info == None or path_info.opt_cost < best_path_info.opt_cost:
            best_path_info = path_info
            time_result.append((total_search_time, float(best_path_info.opt_cost)))

    assert best_path_info != None, "No path found, incorrect config?"
    return (
        best_path_info,
        trials,
        time_result,
    )


def cotengra_hyper_function(
    inputs,
    output,
    size_dict,
    imbalance,
):
    shapes = [tuple([size_dict[i] for i in input]) for input in inputs]
    inputs = [list(input) for input in inputs]
    path_info, _, _ = hybrid_hypercut_greedy(
        inputs, shapes, list(output), imbalances=[imbalance]
    )
    return ctg.ContractionTree.from_path(inputs, output, size_dict, path=path_info.path)


hyper_space = {
    "imbalance": {"type": "FLOAT", "min": 0.01, "max": 0.6},
}
register_hyper_function("hhg", cotengra_hyper_function, hyper_space)
