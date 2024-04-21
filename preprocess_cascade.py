import argparse
import json
import os
from datetime import datetime

from networkx import Graph, DiGraph, read_adjlist, relabel_nodes
from typing import List, Dict, Tuple


def write_cascades(cascade_ids: List[str],
                   trees: Dict[str, List[dict]],
                   file_name: str,
                   user_map: Dict[str, int]):
    with open(file_name, 'w') as f:
        for cid in cascade_ids:
            roots = trees[cid]
            nodes = []
            for root in roots:
                nodes.extend(get_tree_nodes(root, user_map))
            nodes.sort(key=lambda n: n["datetime"])
            times = [node["datetime"] for node in nodes]
            times = [(dt - times[0]).total_seconds() for dt in times]
            nodes = [node["user_id"] for node in nodes]
            line = f'{nodes[0]} {" ".join(f"{nodes[i]} {times[i]}" for i in range(1, len(nodes)))}\n'
            f.write(line)


def create_users_map(graph: Graph) -> dict:
    user_ids = sorted(list(graph.nodes()))
    user_map = {str(user_ids[i]): i for i in range(len(user_ids))}
    return user_map


def limit_nodes(graph: Graph, max_nodes: int):
    """
    Limit the number of nodes of the graph. Remove the nodes with the smallest degrees.
    """
    if graph.number_of_nodes() > max_nodes:
        print("limiting number of nodes ...")
        degrees = {node: graph.degree(node) for node in graph.nodes()}
        nodes = sorted(graph.nodes(), key=lambda node: degrees[node])
        graph.remove_nodes_from(nodes[:graph.number_of_nodes() - max_nodes])


def relabel_graph(graph, new_graph_path, users_map):
    graph = relabel_nodes(graph, users_map)
    with open(new_graph_path, "w") as f:
        f.write(f"{len(users_map)} {graph.number_of_edges()}\n" +
                "\n".join(f"{u} {v}" for u, v in graph.edges()))
    return graph


def get_mapped_node(node, user_map):
    return user_map.setdefault(node, len(user_map))


DT_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_tree_nodes(root: dict, user_map: dict) -> List[str]:
    """
    Get edges of the tree starting from the root. `root` is a `dict` with keys `user_id` and `children`.
    """
    nodes = [{
        "user_id": get_mapped_node(root["user_id"], user_map),
        "datetime": datetime.strptime(root["datetime"], DT_FORMAT)
    }]
    for child in root["children"]:
        nodes.extend(get_tree_nodes(child, user_map))
    return nodes


def get_tree_edges(root: dict, user_map: Dict[str, int]) -> List[Tuple[int, int]]:
    """
    Get edges of the tree starting from the root. `root` is a `dict` with keys `user_id` and `children`.
    """
    if not root["children"]:
        return []
    else:
        root_node = get_mapped_node(root["user_id"], user_map)
        edges = [(root_node, get_mapped_node(child["user_id"], user_map))
                 for child in root["children"]]
        for child in root["children"]:
            edges.extend(get_tree_edges(child, user_map))
        return edges


def write_trees(trees, training, validation, test, out_dir, user_map):
    edges_data = {}
    for cascade_id in trees:
        edges = []
        for root in trees[cascade_id]:
            edges.extend(get_tree_edges(root, user_map))
        edges_data[cascade_id] = edges
    with open(f"{out_dir}/trees-train.json", "w") as f:
        json.dump([edges_data[cid] for cid in training], f)
    with open(f"{out_dir}/trees-val.json", "w") as f:
        json.dump([edges_data[cid] for cid in validation], f)
    with open(f"{out_dir}/trees-test.json", "w") as f:
        json.dump([edges_data[cid] for cid in test], f)


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `Inf-VAE`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-f', '--folds', type=int, default=3, required=True, help="number of cross-validation folds")
    parser.add_argument('-m', '--max_nodes', type=int, required=False, help="maximum number of nodes")
    args = parser.parse_args()

    data_name = args.data
    folds = args.folds

    with open(f'data/{data_name}/graph_info.json') as f:
        graph_info = json.load(f)

    with open(f'data/{data_name}/samples.json') as f:
        samples = json.load(f)

    graph = DiGraph()
    # graph1.txt: directed graph of training and validation
    graph: Graph = read_adjlist(f'data/{data_name}/graph1.txt', create_using=graph).to_undirected()
    if args.max_nodes:
        limit_nodes(graph, args.max_nodes)
    user_map = create_users_map(graph)

    for fold in range(1, folds + 1):
        preprocess_fold(data_name, fold, graph, graph_info, samples, user_map)


def preprocess_fold(data_name, fold, graph, graph_info, samples, user_map):
    out_dir = f'data/{data_name}/{data_name}-{fold}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    training = graph_info[f'graph{fold + 1}']
    validation = list(set(samples['training']) - set(training))
    test = samples['test']

    print(f"=== fold #{fold} ===")
    print(f'training length: {len(training)}')
    print(f'validation length: {len(validation)}')
    print(f'test length: {len(test)}')

    with open(f'data/{data_name}/trees.json') as f:
        trees = json.load(f)

    print("writing trees to files ...")
    write_trees(trees, training, validation, test, out_dir, user_map)

    print("writing cascades to files ...")
    write_cascades(training, trees, f'{out_dir}/train.txt', user_map)
    write_cascades(validation, trees, f'{out_dir}/val.txt', user_map)
    write_cascades(test, trees, f'{out_dir}/test.txt', user_map)

    print("writing graph ...")
    relabel_graph(graph, f'{out_dir}/graph.txt', user_map)


if __name__ == '__main__':
    main()
