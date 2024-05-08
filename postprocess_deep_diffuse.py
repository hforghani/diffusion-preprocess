import argparse
import json
import os

from metric import report_evals
from models import Result
from prediction import predict_edges


def load_seen_nodes(data_path):
    node_file = f"{data_path}/seen_nodes.txt"
    with open(node_file, 'r') as f:
        seen_nodes = [int(x.strip()) for x in f]
    return seen_nodes


def read_results(log_file, data_dir):
    # Read the log file.
    with open(log_file) as f:
        results = json.load(f)

    seen_nodes = load_seen_nodes(data_dir)
    sequences = [[seen_nodes[index] for index in row] for row in results["sequences"]]
    targets = [[seen_nodes[index] for index in row] for row in results["targets"]]
    outputs = [[seen_nodes[index] for index in row] for row in results["outputs"]]
    results = [Result(seq, out, target) for seq, target, out in zip(sequences, targets, outputs)]
    return results


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `Inf-VAE`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-l', '--log-file', required=True, help="Inf-VAE log file")
    parser.add_argument('--frequency', type=int, default=10, help="evaluation frequency")
    args = parser.parse_args()

    data_dir = os.path.join("data/deep-diffuse", args.data)
    print("reading log file ...")
    results = read_results(args.log_file, data_dir)
    print("predicting edges and evaluating ...")
    data_path = f"data/deep-diffuse/{args.data}"
    evals = predict_edges(results, args.data, data_path, eval_freq=args.frequency)
    report_evals(evals, "deepdiffuse", "edges", args.data, "results/deep-diffuse")


if __name__ == '__main__':
    main()
