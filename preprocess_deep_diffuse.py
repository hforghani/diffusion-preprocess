import argparse
import os.path
import shutil

from networkx import read_adjlist


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


def preprocess_fold(data_name, fold_num):
    infvae_dir = f"data/inf-vae/{data_name}-{fold_num}"
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
    deep_diffuse_dir = f"data/deep-diffuse/{data_name}-{fold_num}"
    if not os.path.exists(deep_diffuse_dir):
        os.mkdir(deep_diffuse_dir)

    with open(f"{deep_diffuse_dir}/train.txt", "w") as f:
        f.write(train_content)
    with open(f"{deep_diffuse_dir}/test.txt", "w") as f:
        f.write(test_content)

    shutil.copyfile(f"{infvae_dir}/graph.txt", f"{deep_diffuse_dir}/graph.txt")

    graph = read_adjlist(f"{infvae_dir}/graph.txt")
    graph.add_nodes_from(read_nodes(test_content))
    graph.add_nodes_from(read_nodes(val_content))
    with open(f"{deep_diffuse_dir}/seen_nodes.txt", "w") as f:
        f.write("\n".join(graph.nodes()))


def main():
    parser = argparse.ArgumentParser('Process data of `Inf-VAE` in order to fed into `deep-diffuse`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    args = parser.parse_args()

    data_name = args.data

    for fold_num in range(1, 4):
        preprocess_fold(data_name, fold_num)


if __name__ == '__main__':
    main()
