import json
import os
from typing import List, Tuple

from networkx import DiGraph, read_adjlist, relabel_nodes

from metric import Metric
from models import Result


def get_subtree_edges(root: dict) -> List[Tuple[int, int]]:
    """
    Get edges of the tree starting from the root. `root` is a `dict` with keys `user_id` and `children`.
    """
    if not root["children"]:
        return []
    else:
        edges = [(root["user_id"], child["user_id"]) for child in root["children"]]
        for child in root["children"]:
            edges.extend(get_subtree_edges(child))
        return edges


def get_tree_edges(trees):
    edges_data = {}
    for cascade_id in trees:
        edges = []
        for root in trees[cascade_id]:
            edges.extend(get_subtree_edges(root))
        edges_data[cascade_id] = edges
    return edges_data


# def get_target_edges(result, edges_data, test_inputs):
#     """
#     Search in the test inputs. Find the cascade with the equal input nodes and return its edges data.
#     """
#     for cid in test_inputs:
#         if set(test_inputs) == set(result.inputs):
#             return edges_data[cid]
#     else:
#         raise ValueError(f"cascade not found")

# def get_target_edges(result, trees):
#     """
#     Search in the test inputs. Find the cascade with the equal input nodes and return its edges data.
#     """
#     targets = set(result.targets)
#     for tree in trees:
#         tree_targets = {edge[1] for edge in tree}
#         if tree_targets == targets:
#             return [tuple(edge) for edge in tree]
#     else:
#         raise ValueError(f"cascade not found")


def predict_edges(results: List[Result], data_name: str, data_path: str, eval_freq=10) -> List[List[Metric]]:
    """
    For each result in `results` starts with initial nodes of `result.inputs`, add nodes in `result.outputs` one
    by one and put the best matching edge. At each node adding, evaluates the current result. Finally, for each
    result we have a list of evaluations. Evaluation result lists must be of the same lengths.
    """
    graph = DiGraph()
    graph = read_adjlist(os.path.join("data/diffusion", data_name, "graph-dir.txt"), create_using=graph)
    graph = relabel_nodes(graph, {node: int(node) for node in graph})
    evaluations = []

    with open(os.path.join(data_path, "trees-test.json")) as f:
        trees = json.load(f)

    test_size = min(len(trees), len(results))

    # Consider length of results may be larger or smaller than test_size in `deep-diffuse` due to its implementation.
    # So we only iterate on its first `test_size` elements.
    for i in range(test_size):
        print(f"evaluating cascade {i} ...")
        res = results[i]
        target_edges = {tuple(edge) for edge in trees[i]}
        initial_edges = set()  # Since initial depth = 0
        ref_set = set(graph.edges()) | target_edges - initial_edges
        active_nodes = res.inputs.copy()
        successors = {node: set(graph.successors(node)) if node in graph else set() for node in active_nodes}
        edges = []
        res_eval = []

        # At each iteration evaluate the result of top `k` output.
        for k in range(len(res.outputs)):
            node = res.outputs[k]
            if node not in active_nodes:
                # Find the parent node and add the new edge.
                for active_node in successors:
                    if node in successors[active_node]:
                        edges.append((active_node, node))
                        active_nodes.append(node)
                        successors[node] = set(graph.successors(node))
                        break

            if k % eval_freq == 0:
                # Evaluate the result of the top k output.
                metric = Metric(edges, target_edges, ref_set)
                res_eval.append(metric)

        evaluations.append(res_eval)

    return evaluations
