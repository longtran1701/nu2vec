import argparse
import scipy.spatial.distance as dist
import numpy as np

"""
Parses the arguments into an arguments object.
"""
def parse_args():
    p = argparse.ArgumentParser(description="Function prediction.")
    p.add_argument("network", help="Network file.")
    p.add_argument("labels", help="Node labels file.")
    p.add_argument("--knn", type=int, help="k-nearest neighbors vote.")
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

if __name__ == "__main__":
    args = parse_args()

    labels = parse_labels(args.labels)
    labelling = None

    if args.mv:
        print("Majority Vote")
    elif args.wmv:
        print("Weighted Majority Vote")
    elif args.knn is not None:
        (mat, nna) = parse_embedding(args.network)
        labelling = knn(mat, nna, labels, args.knn)

    """ Print labelling to stdout """
    for node, labels in labelling.items():
        labels_str = " ".join(labels)
        print(node + " " + labels_str)

    exit(0)
