import argparse
import json
import logging
import os
import re
import numpy as np

from networkx import DiGraph, read_adjlist, relabel_nodes

from metric import Metric
from roc import report_and_save_roc


def read_results(content, data_dir):
    with open(os.path.join(data_dir, "test.txt")) as f:
        test_count = len(f.readlines())

    with open(os.path.join(data_dir, "graph.txt")) as f:
        num_nodes = int(f.readline().split()[0])

    # Read inputs and targets from the log file.
    inputs = re.findall(r"(\d+)\s+input: \[(.+)\]", content)
    targets = re.findall(r"\s+target: \[(.+)\]", content)

    # Get data of the last run.
    indexes = [item[0] for item in inputs]
    last_zero = len(indexes) - indexes[::-1].index("0") - 1
    inputs = inputs[last_zero:]
    targets = targets[last_zero:]
    results = [
        {
            "input": [int(node) for node in inputs[i][1].split(", ")],
            "target": [int(node) for node in targets[i].split(", ")],
        }
        for i in range(test_count)
    ]

    # Read epoch results from log file.
    epoch_inputs = re.findall(r"inputs\[(\d+)\] = \[ (.+) \]", content)
    epoch_outputs = re.findall(r"output_filter\[(\d+)\] = \[ (.+) \]", content)
    epoch_targets = re.findall(r"targets\[(\d+)\] = \[ (.+) \]", content)
    # rel_scores = re.findall(r"output_relevance_score\[(\d+)\] = (\S+)", content)

    # Get the result of the last epoch.
    indexes = [item[0] for item in epoch_inputs]
    last_zero = len(indexes) - indexes[::-1].index("0") - 1
    epoch_inputs = epoch_inputs[last_zero:]
    epoch_outputs = epoch_outputs[last_zero:]
    epoch_targets = epoch_targets[last_zero:]
    # rel_scores = rel_scores[last_zero:]
    epoch_results = [
        {
            "input": [int(node) for node in epoch_inputs[i][1].split(", ") if node != str(num_nodes)],
            "output": [int(node) for node in epoch_outputs[i][1].split(", ") if node != "-1"],
            "target": [int(node) for node in epoch_targets[i][1].split(", ") if node != "-1"],
            # "rel_score": [int(d) if d in "01" else -1 for d in rel_scores[i][1]],
        }
        for i in range(len(epoch_inputs))
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
            logging.warning("no equal input and target found")
            results_to_remove.append(res)

    # Remove the results for which the inputs and targets have not been found.
    for res in results_to_remove:
        results.remove(res)

    # print(results)
    return results


def predict_edges(results, data_name, fold_num):
    graph = DiGraph()
    graph = read_adjlist(os.path.join("data", data_name, "graph-dir.txt"), create_using=graph)
    graph = relabel_nodes(graph, {node: int(node) for node in graph})
    evaluations = []

    with open(os.path.join("data", data_name, f"{data_name}-{fold_num}", "trees-test.json")) as f:
        trees = json.load(f)

    for i in range(len(results)):
        res = results[i]
        target_edges = {tuple(edge) for edge in trees[i]}
        active_nodes = res["input"].copy()
        successors = {node: list(graph.successors(node)) if node in graph else [] for node in active_nodes}
        edges = []
        res_eval = []

        # At each iteration evaluate the result of top `k` output.
        for k in range(len(res["output"])):
            node = res["output"][k]

            # Find the parent node and add the new edge.
            for active_node in successors:
                if node in successors[active_node]:
                    edges.append((active_node, node))
                    active_nodes.append(node)
                    successors[node] = list(graph.successors(node))
                    break

            # Evaluate the result of top k output.
            initial_edges = set()  # Since initial depth = 0
            ref_set = set(graph.edges()) | set(target_edges) - initial_edges
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

    fprs = [np.array([eval[k]["fpr"] for eval in evals]).mean() for k in range(max_k)]
    tprs = [np.array([eval[k]["tpr"] for eval in evals]).mean() for k in range(max_k)]
    report_and_save_roc(fprs, tprs, "infvae", "edges", data_name)


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `Inf-VAE`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-f', '--fold', required=True, help="fold number")
    parser.add_argument('-l', '--log-file', required=True, help="Inf-VAE log file")
    args = parser.parse_args()

    with open(args.log_file) as f:
        content = f.read()

    data_dir = os.path.join("data", args.data, f"{args.data}-{args.fold}")
    print("reading log file ...")
    results = read_results(content, data_dir)
    print("predicting edges and evaluating ...")
    evals = predict_edges(results, args.data, args.fold)
    report_evals(evals, args.data)


if __name__ == '__main__':
    main()
