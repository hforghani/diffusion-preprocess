import argparse
import json
import os
import re
import warnings

import numpy as np

from networkx import DiGraph, read_adjlist, relabel_nodes

from metric import Metric
from roc import report_and_save_roc


def read_results(lines, data_dir):
    with open(os.path.join(data_dir, "graph.txt")) as f:
        num_nodes = int(f.readline().split()[0])

    # Read the log file.
    inputs, targets = [], []  # input and target of test set
    epoch_inputs, epoch_outputs, epoch_targets = [], [], []  # inputs, targets and predicted outputs for all epochs
    for line in lines:
        match = re.match(r"(\d+)\s+input: (\[.+\])", line)
        if match:
            inputs.append(match.groups())
        else:
            match = re.match(r"\s+target: (\[.+\])", line)
            if match:
                targets.append(match.groups())
            else:
                match = re.match(r"inputs\[(\d+)\] = (\[ .+ \])", line)
                if match:
                    epoch_inputs.append(match.groups())
                else:
                    match = re.match(r"outputs\[(\d+)\] = (\[ .+ \])", line)
                    if match:
                        epoch_outputs.append(match.groups())
                    else:
                        match = re.match(r"targets\[(\d+)\] = (\[ .+ \])", line)
                        if match:
                            epoch_targets.append(match.groups())

    # Get data of the last run.
    indexes = [item[0] for item in inputs]
    last_zero = len(indexes) - indexes[::-1].index("0") - 1
    inputs = inputs[last_zero:]
    targets = targets[last_zero:]
    results = [
        {
            "input": [int(node) for node in eval(inputs[i][1])],
            "target": [int(node) for node in eval(targets[i][0])],
        }
        for i in range(len(inputs))
    ]
    test_count = len(results)

    # Peek 2 * test_count last results in order to ensure cover all test examples.
    # `epoch_results` are in reverse order.
    epoch_results = [
        {
            "input": [int(node) for node in eval(epoch_inputs[-i][1]) if node != num_nodes],
            "output": [int(node) for node in eval(epoch_outputs[-i][1]) if node != -1],
            "target": [int(node) for node in eval(epoch_targets[-i][1]) if node != -1],
        }
        for i in range(1, 2 * test_count)
    ]

    # Since the order of epoch results is not same as the original test set, find the outputs by comparing inputs
    # and targets.
    results_to_remove = []
    for res in results:
        for epoch_res in epoch_results:
            if res["input"] == epoch_res["input"] and res["target"] == epoch_res["target"]:
                res["output"] = epoch_res["output"]
                break
        else:
            warnings.warn("no equal input and target found")
            results_to_remove.append(res)

    # Remove the results for which the inputs and targets have not been found.
    for res in results_to_remove:
        results.remove(res)

    # print(results)
    return results


def predict_edges(results, data_name, fold_num):
    graph = DiGraph()
    graph = read_adjlist(os.path.join("data/diffusion", data_name, "graph-dir.txt"), create_using=graph)
    graph = relabel_nodes(graph, {node: int(node) for node in graph})
    evaluations = []

    with open(os.path.join("data/inf-vae", f"{data_name}-{fold_num}", "trees-test.json")) as f:
        trees = json.load(f)

    for i in range(len(results)):
        # for i in range(10):
        print(f"evaluating cascade {i} ...")
        res = results[i]
        target_edges = {tuple(edge) for edge in trees[i]}
        active_nodes = res["input"].copy()
        successors = {node: set(graph.successors(node)) if node in graph else set() for node in active_nodes}
        edges = []
        res_eval = []

        # At each iteration evaluate the result of top `k` output.
        for k in range(len(res["output"])):
            node = res["output"][k]
            if node not in active_nodes:
                # Find the parent node and add the new edge.
                for active_node in successors:
                    if node in successors[active_node]:
                        edges.append((active_node, node))
                        active_nodes.append(node)
                        successors[node] = set(graph.successors(node))
                        break

            if k % 10 == 0:
                # Evaluate the result of top k output.
                initial_edges = set()  # Since initial depth = 0
                ref_set = set(graph.edges()) | target_edges - initial_edges
                metric = Metric(edges, target_edges, ref_set)
                res_eval.append(metric)

        evaluations.append(res_eval)

    return evaluations


def report_evals(evals, data_name):
    max_k = len(evals[0])
    mean_f1 = []

    for k in range(max_k):
        f1s_k = [eval[k]["f1"] for eval in evals]
        mean_f1.append(np.array(f1s_k).mean())

    max_f1 = np.max(np.array(mean_f1))
    print(f"Max F1 for edges: {max_f1}")

    fprs = [np.array([cas_eval[k]["fpr"] for cas_eval in evals]).mean() for k in range(max_k)]
    tprs = [np.array([cas_eval[k]["tpr"] for cas_eval in evals]).mean() for k in range(max_k)]
    report_and_save_roc(fprs, tprs, "infvae", "edges", data_name)


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `Inf-VAE`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-f', '--fold', required=True, help="fold number")
    parser.add_argument('-l', '--log-file', required=True, help="Inf-VAE log file")
    args = parser.parse_args()

    with open(args.log_file) as f:
        lines = f.readlines()

    data_dir = os.path.join("data/inf-vae", f"{args.data}-{args.fold}")
    print("reading log file ...")
    results = read_results(lines, data_dir)
    print("predicting edges and evaluating ...")
    evals = predict_edges(results, args.data, args.fold)
    report_evals(evals, args.data)


if __name__ == '__main__':
    main()
