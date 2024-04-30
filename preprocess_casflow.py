import argparse
import json
from datetime import datetime, timedelta

from networkx import Graph, DiGraph, read_adjlist, relabel_nodes, write_adjlist
from typing import Dict

DT_FORMAT = '%Y-%m-%d %H:%M:%S'


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


def write_graph(graph, new_graph_path):
    with open(new_graph_path, "w") as f:
        f.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n" +
                "\n".join(f"{u} {v}" for u, v in graph.edges()))
    return graph


def get_mapped_node(node, user_map):
    return user_map.setdefault(node, len(user_map))


DT_FORMAT = '%Y-%m-%d %H:%M:%S'


def get_tree_edges(root: dict, user_map: Dict[str, int]):
    """
    Get edges of the tree starting from the root. `root` is a `dict` with keys `user_id` and `children`.
    """
    if not root["children"]:
        return []
    else:
        root_node = get_mapped_node(root["user_id"], user_map)
        edges = [
            [  # [(node1, node2), time]
                (root_node,
                 get_mapped_node(child["user_id"], user_map)),
                child["datetime"]
            ]
            for child in root["children"]
        ]
        for child in root["children"]:
            edges.extend(get_tree_edges(child, user_map))
        return edges


def extract_edges(tree, first_time, user_map):
    edges = [
        [  # [(node1, node2), time]
            (get_mapped_node("root", user_map),
             get_mapped_node(root["user_id"], user_map)),
            first_time
        ] for root in tree
    ]
    for root in tree:
        edges.extend(get_tree_edges(root, user_map))

    # Convert datetime strings to total seconds since the first activation.
    for edge in edges:
        edge[1] = int((edge[1] - first_time).total_seconds())

    return edges


def correct_times(tree):
    def convert_time(parent):
        parent["datetime"] = datetime.strptime(parent["datetime"], DT_FORMAT)
        for child in parent["children"]:
            convert_time(child)

    def check_parent_precedence(parent):
        for child in parent["children"]:
            if child["datetime"] < parent["datetime"]:
                child["datetime"] = parent["datetime"] + timedelta(seconds=60)

    for root in tree:
        convert_time(root)
        check_parent_precedence(root)


def main():
    parser = argparse.ArgumentParser('Process data of `diffusion` code in order to fed into `CasFlow`')
    parser.add_argument('-d', '--data', required=True, help="data directory name")
    parser.add_argument('-m', '--max_nodes', type=int, required=False, help="maximum number of nodes")
    args = parser.parse_args()

    data_name = args.data

    with open(f'data/diffusion/{data_name}/graph_info.json') as f:
        graph_info = json.load(f)

    with open(f'data/diffusion/{data_name}/samples.json') as f:
        samples = json.load(f)

    # Find the graph related to all training data.
    all_train_graph_name = None
    for graph_name in graph_info:
        if set(graph_info[graph_name]) == set(samples['training']):
            all_train_graph_name = graph_name
            break
    assert all_train_graph_name is not None

    # Read the graph file.
    graph_dir = DiGraph()
    graph_dir = read_adjlist(f'data/diffusion/{data_name}/{all_train_graph_name}.txt', create_using=graph_dir)

    # Since in CasFlow cascades must be trees (with only 1 root), add a `root` node having an edge to each node in the
    # graph. Also for each cascade, add edges from the `root` node to all cascade roots.
    graph_dir.add_edges_from([("root", node) for node in graph_dir])

    user_map = create_users_map(graph_dir)  # Map mongodb user ids to int
    graph_dir = relabel_nodes(graph_dir, user_map)
    write_adjlist(graph_dir, f"data/diffusion/{data_name}/graph-dir-aug-root.txt")

    with open(f'data/diffusion/{data_name}/trees.json') as f:
        trees = json.load(f)

    root = get_mapped_node("root", user_map)
    with open(f"data/casflow/{data_name}/dataset.txt", "w") as f:
        for cascade_id, tree in trees.items():
            correct_times(tree)
            first_time = min(root["datetime"] for root in tree)
            first_time_ord = first_time.toordinal()
            edges = extract_edges(tree, first_time, user_map)
            edges_data = " ".join(f"{edge[0][0]}/{edge[0][1]}:{edge[1]}" for edge in edges)
            paths = f"{root}:0 {edges_data}"
            f.write(f"{cascade_id}\t0\t{first_time_ord}\t{len(edges) + 1}\t{paths}\n")


if __name__ == '__main__':
    main()
