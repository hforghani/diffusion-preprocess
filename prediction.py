import json
import os
import random
from typing import List

from networkx import DiGraph, read_adjlist, relabel_nodes

from metric import Metric
from models import Result


def predict_edges(results: List[Result], data_name: str, data_path: str, eval_freq=10) -> List[List[Metric]]:
    graph = DiGraph()
    graph = read_adjlist(os.path.join("data/diffusion", data_name, "graph-dir.txt"), create_using=graph)
    # graph = read_adjlist(f"data/deep-diffuse/{data_name}/graph.txt", create_using=graph)
    graph = relabel_nodes(graph, {node: int(node) for node in graph})
    max_out_len = max(len(res.outputs) for res in results)
    evaluations = []

    with open(os.path.join(data_path, "trees-test.json")) as f:
        trees = json.load(f)
    # trees = [[(random.randint(1, 10000), random.randint(1, 10000)) for _ in range(100)] for i in range(len(results))]
    test_size = len(trees)

    # Consider length of results may be greater than test_size in `deep-diffuse` due to its implementation. So
    # we only iterate on its first `test_size` elements.

    for i in range(test_size):
        # for i in range(10):
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
                # Evaluate the result of top k output.
                metric = Metric(edges, target_edges, ref_set)
                res_eval.append(metric)

        # Fill the rest with the last evaluation resuls.
        last_eval = res_eval[-1]
        for k in range(max_out_len - len(res.outputs)):
            res_eval.append(Metric.from_values(**last_eval.metrics))

        evaluations.append(res_eval)

    return evaluations
