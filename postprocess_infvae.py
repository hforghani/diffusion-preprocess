import argparse
import os
import re
import warnings

from metric import report_evals
from models import Result
from prediction import predict_edges


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
        Result(
            inputs=[int(node) for node in eval(inputs[i][1])],
            outputs=None,
            targets=[int(node) for node in eval(targets[i][0])]
        )
        for i in range(len(inputs))
    ]
    test_count = len(results)

    # Peek 2 * test_count last results in order to ensure cover all test examples.
    # `epoch_results` are in reverse order.
    epoch_results = [
        Result(
            inputs=[int(node) for node in eval(epoch_inputs[-i][1]) if node != num_nodes],
            outputs=[int(node) for node in eval(epoch_outputs[-i][1]) if node != -1],
            targets=[int(node) for node in eval(epoch_targets[-i][1]) if node != -1]
        )
        for i in range(1, 2 * test_count)
    ]

    # Since the order of epoch results is not same as the original test set, find the outputs by comparing inputs
    # and targets.
    results_to_remove = []
    for res in results:
        for epoch_res in epoch_results:
            if res.inputs == epoch_res.inputs and res.targets == epoch_res.targets:
                res.outputs = epoch_res.outputs
                break
        else:
            warnings.warn("no equal input and target found")
            results_to_remove.append(res)

    # Remove the results for which the inputs and targets have not been found.
    for res in results_to_remove:
        results.remove(res)

    # print(results)
    return results


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `Inf-VAE`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-f', '--fold', required=True, help="fold number")
    parser.add_argument('-l', '--log-file', required=True, help="Inf-VAE log file")
    parser.add_argument('--frequency', type=int, default=10, help="evaluation frequency")
    args = parser.parse_args()

    with open(args.log_file) as f:
        lines = f.readlines()

    data_dir = os.path.join("data/inf-vae", f"{args.data}-{args.fold}")
    print("reading log file ...")
    results = read_results(lines, data_dir)
    print("predicting edges and evaluating ...")
    data_path = os.path.join("data/inf-vae", f"{args.data}-{args.fold}")
    evals = predict_edges(results, args.data, data_path, eval_freq=args.frequency)
    report_evals(evals, "infvae", "edges", args.data, "results/inf-vae")


if __name__ == '__main__':
    main()
