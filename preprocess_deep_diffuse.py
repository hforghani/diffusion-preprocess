import argparse
import json
import os.path
import shutil

from networkx import read_adjlist, Graph


def add_first_time(content):
    new_content = ""
    i = 0
    for line in content.split("\n"):
        line = line.strip()
        if line:
            first_node, rest = line.split(" ", maxsplit=1)
            rest = " ".join(str(int(float(num))) for num in rest.split(" "))
            new_content += f"{i} {first_node} 0 {rest}\n"
            i += 1
    return new_content


def read_nodes(content):
    nodes = set()
    for line in content.split("\n"):
        line = line.strip()
        if line:
            _, rest = line.split(" ", maxsplit=1)
            nodes.update(set(rest.split(" ")[0::2]))
    return nodes


def preprocess(data_name):
    infvae_dir = f"data/inf-vae/{data_name}-1"
    with open(f"{infvae_dir}/train.txt") as f:
        train_content = f.read()
    with open(f"{infvae_dir}/val.txt") as f:
        val_content = f.read()
    with open(f"{infvae_dir}/test.txt") as f:
        test_content = f.read()

    train_content = add_first_time(train_content)
    val_content = add_first_time(val_content)
    test_content = add_first_time(test_content)

    train_content += val_content

    if not os.path.exists("data/deep-diffuse"):
        os.mkdir("data/deep-diffuse")
    deep_diffuse_dir = f"data/deep-diffuse/{data_name}"
    if not os.path.exists(deep_diffuse_dir):
        os.mkdir(deep_diffuse_dir)

    with open(f"{deep_diffuse_dir}/train.txt", "w") as f:
        f.write(train_content)
    with open(f"{deep_diffuse_dir}/test.txt", "w") as f:
        f.write(test_content)

    shutil.copyfile(f"{infvae_dir}/graph.txt", f"{deep_diffuse_dir}/graph.txt")
    shutil.copyfile(f"{infvae_dir}/trees-test.json", f"{deep_diffuse_dir}/trees-test.json")

    graph: Graph = read_adjlist(f"{infvae_dir}/graph.txt")
    graph.add_nodes_from(read_nodes(test_content))
    graph.add_nodes_from(read_nodes(val_content))
    # Put a fake node at the beginning in order to neglect index 0 due to a bug in deep-diffuse code.
    nodes = [str(graph.number_of_nodes())] + list(graph.nodes())

    with open(f"{deep_diffuse_dir}/seen_nodes.txt", "w") as f:
        f.write("\n".join(nodes))

    write_seed_counts(data_name, deep_diffuse_dir)


def write_seed_counts(data_name, deep_diffuse_dir):
    with open(f"data/diffusion/{data_name}/trees.json") as f:
        trees = json.load(f)
    with open(f"data/diffusion/{data_name}/samples.json") as f:
        samples = json.load(f)
    seeds = [len(trees[cid]) for cid in samples["test"]]  # Number of roots (seed nodes)
    with open(f"{deep_diffuse_dir}/seed_counts.txt", "w") as f:
        f.write("\n".join(str(num) for num in seeds))


def main():
    parser = argparse.ArgumentParser('Process data of `Inf-VAE` in order to fed into `deep-diffuse`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    args = parser.parse_args()

    data_name = args.data

    preprocess(data_name)


if __name__ == '__main__':
    main()
