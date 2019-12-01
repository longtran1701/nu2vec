import argparse
import scipy.spatial.distance as dist
import numpy as np
import networkx as nx

"""
Current TODOs:
  - Only chooses best label, output list of l abels with significance
"""

"""
Parses the arguments into an arguments object.
"""
def parse_args():
    p = argparse.ArgumentParser(description="Function prediction.")
    p.add_argument("network", help="Network file.")
    p.add_argument("labels", help="Node labels file.")
    p.add_argument("--knn", type=int, help="k-nearest neighbors vote.")
    p.add_argument("--string", type=int,
                   help="Weighted majority vote on STRING data.")
    p.add_argument("--mv", action="store_true",
                   help="Standard majority vote.")
    p.add_argument("--wmv", action="store_true",
                   help="Weighted majority vote.")
    return p.parse_args()

"""
Takes a node-labels text file and outputs a
dictionary mapping label node name to a list
of labels.
"""
def parse_labels(fname):
    node_name_to_labels = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            name = words[0]
            labels = words[1:]
            node_name_to_labels[name] = labels

    return node_name_to_labels

"""
Takes a node2vec embedding file and outputs
both a NxD matrix and an array where the ith
entry contains the name of the node in the ith
row of the matrix.
"""
def parse_embedding(fname):
    with open(fname, "r") as f:
        lines = f.readlines()

        dimensions = lines[0].split()
        n = int(dimensions[0])
        d = int(dimensions[1])

        matrix = np.empty([n, d])
        node_name_array = [None] * n
        for i in range(n):
            line = lines[i + 1]
            words = line.split()

            name = words[0]
            values = list(map(float, words[1:]))
            matrix[i] = values
            node_name_array[i] = name

        return (matrix, node_name_array)

"""
Parses the STRING network file into a weighted networkx
graph using the specified column as the weights.

Reference file format is 4932.protein.links.detailed.v11.0.txt
"""
def parse_string_network(fname, column):
    graph = nx.Graph()

    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            words = line.split()

            protein1 = words[0][5:]
            protein2 = words[1][5:]
            weight = float(words[column + 2])

            graph.add_edge(protein1, protein2, weight=weight)

    return graph

"""
Returns most popular label among the voters,
optionally weighted by their significance.
"""
def vote(voters, labels, weights=None):
    label_counts = {}

    for voter in voters:
        for label in labels[voter]:
            weight = 1
            if weights is not None:
                weight = weights[voter]

            if label not in label_counts:
                label_counts[label] = weight
            else:
                label_counts[label] += weight

    return max(label_counts, key=lambda k: label_counts[k])

"""
Uses k-nearest neighbors to vote for
the label on unlabeled nodes.

Outputs a labelling for the nodes.
"""
def knn(matrix, node_names, labels, k):
    labelling = {}

    distances = dist.squareform(dist.pdist(matrix))

    for i in range(len(matrix)):
        node = node_names[i]

        if node in labels:
            labelling[node] = labels[node]
            continue

        voter_ids = np.argsort(distances[i])[1:(k + 1)]
        voters = [node_names[i] for i in voter_ids]
        label = vote(voters, labels)
        labelling[node] = [label]

    return labelling

"""
Performs majority vote on unlabeled nodes
in the graph.

Outputs a labelling for the nodes.
"""
def mv(G, labels, weighted=False):
    labelling = {}

    for node in G.nodes():
        if node in labels:
            labelling[node] = labels[node]
            continue

        voters = list(G[node])
        if weighted:
            weights = {}

            for (voter, data) in G[node].items():
                weights[voter] = data["weight"]

            label = vote(voters, labels, weights=weights)
            labelling[node] = [label]
        else:
            label = vote(voters, labels)
            labelling[node] = [label]

    return labelling

if __name__ == "__main__":
    args = parse_args()

    labels = parse_labels(args.labels)
    labelling = None

    if args.mv:
        graph = nx.readwrite.edgelist.read_edgelist(args.network)
        labelling = mv(graph, labels, weighted=False)
    elif args.wmv:
        graph = nx.readwrite.read_weighted_edgelist(args.network)
        labelling = mv(graph, labels, weighted=True)
    elif args.knn is not None:
        (mat, nna) = parse_embedding(args.network)
        labelling = knn(mat, nna, labels, args.knn)
    elif args.string is not None:
        graph = parse_string_network(args.network, args.string)
        labelling = mv(graph, args.labels, weighted=True)

    """ Print labelling to stdout """
    for node, labels in labelling.items():
        labels_str = " ".join(labels)
        print(node + " " + labels_str)

    exit(0)
