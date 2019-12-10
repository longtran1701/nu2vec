import argparse
import scipy.spatial.distance as dist
import numpy as np
import networkx as nx
import random
from collections import defaultdict


"""
Current TODOs:
  - Only chooses best label, output list of l abels with significance
  - Argument parsing is a code smell. I don't like it.
  - Investigate potential lack of determinism
  - TRIPLE check cross validation code.
"""

"""
Parses the arguments into an arguments object.
"""
def parse_args():
    p = argparse.ArgumentParser(description="Function prediction.")
    p.add_argument("network", help="Network file.")
    p.add_argument("labels", help="Node labels file.")
    p.add_argument("--network-type", "-t", required=True,
                   choices=["edgelist", "weighted_edgelist",
                            "embedding", "string"],
                   help="Type of network file.")
    p.add_argument("--algorithm", "-a", required=True,
                   choices=["mv", "wmv", "knn"],
                   help="Function prediction algorithm to use.")
    p.add_argument("--cross-validate", type=int,
                   help="Assumes label list is full and performs k-fold "
                        "cross validation.")
    p.add_argument("--args", nargs='*',
                   help="Extra arguments for prediction algorithm.")
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
            if labels:
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
        print(lines[0].split()[column])
        for line in lines[1:]:
            words = line.split()

            protein1 = words[0].split('.')
            protein1 = protein1[0] if len(protein1) == 1 else protein1[1]
            protein2 = words[1].split('.')
            protein2 = protein2[0] if len(protein2) == 1 else protein2[1]
            weight = float(words[column])

            if weight != 0:
                graph.add_edge(protein1, protein2, weight=weight)
    
    print(len(graph))
    return graph

"""
Parses the network file using the appropriate method.
"""
def parse_network(args):
    if args.network_type == "edgelist":
        return nx.readwrite.edgelist.read_edgelist(args.network)
    elif args.network_type == "weighted_edgelist":
        return nx.readwrite.edgelist.read_weighted_edgelist(args.network)
    elif args.network_type == "embedding":
        return parse_embedding(args.network)
    elif args.network_type == "string":
        column = int(args.args[0])
        return parse_string_network(args.network, column)

"""
Returns most popular label among the voters,
optionally weighted by their significance.

Requires each voter to be labeled.
"""
def vote(voters, labels, weights=None):
    label_counts = defaultdict(int)
    for voter in voters:
        for label in labels[voter]:
            label_counts[label] += (1 if not weights else weights[voter])
    
    if not label_counts:
        return random.choice(list(set(np.array(labels.values()).flatten())))

    return max(label_counts.keys(), key=lambda k: label_counts[k])

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

        """ If node is already labeled, don't label it again! """
        if node in labels:
            labelling[node] = labels[node]
            continue

        sorted_voter_ids = np.argsort(distances[i])[1:]

        voters = []
        weights = {}
        j = 0
        while len(voters) < k and j < len(sorted_voter_ids):
            potential_voter_id = sorted_voter_ids[j]
            potential_voter = node_names[potential_voter_id]
            if potential_voter in labels:
                voters.append(potential_voter)
                weights[potential_voter] = 1. / float(distances[i][potential_voter_id])
            j += 1

        label = vote(voters, labels, weights)
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

        voters = filter(lambda x: x in labels, G[node])

        weights = None
        if weighted:
            weights = {voter : data["weight"] for (voter, data) in G[node].items()}

        label = vote(voters, labels, weights=weights)
        labelling[node] = [label]

    return labelling

"""
Runs algorithm, returns labelling.
"""
def run_algorithm(network, labels, args):
    if args.algorithm == "mv":
        return mv(network, labels, weighted=False)
    elif args.algorithm == "wmv":
        return mv(network, labels, weighted=True)
    elif args.algorithm == "knn":
        try:
            k = int(args.args[0])
        except TypeError:
            print("Expected argument for k-nearest neighbors.")
            exit(1)
        except ValueError:
            print("Expected integer argument.")
            exit(1)
        
        (mat, nna) = tuple(network)
        return knn(mat, nna, labels, k)


"""
Scores cross validation by counting the
number of test nodes that were accurately labeled
after their removal from the true labelling.
"""
def score_cv(test_nodes, test_labelling, real_labelling):
    correct = 0
    total = 0
    for node in test_nodes:

        # ignore nodes that are unlabelled in training set
        if node not in test_labelling:
            continue

        test_label = test_labelling[node][0]
        if test_label in real_labelling[node]:
            correct += 1
        total += 1

    return float(correct) / float(total)

if __name__ == "__main__":
    args = parse_args()
    labels = parse_labels(args.labels)
    network = parse_network(args)
    print(f'There are {len(labels.keys())} labeled nodes')

    random.seed(0)

    """ In cross validation, labels are
    assumed to cover every node. """
    if args.cross_validate is not None:
        nodes = list(labels.keys())  # only look at nodes with labels
        random.shuffle(nodes)
        accuracies = []
        """ Remove n / k nodes from labelling and run algorithm on
            each set. """
        from tqdm import tqdm
        for i in tqdm(range(0, args.cross_validate)):
            inc = int(len(nodes) / args.cross_validate)

            x = inc * i
            y = inc * (i + 1)
            if i + 1 == args.cross_validate:
                y = len(nodes)

            training_nodes = nodes[:x] + nodes[y:]
            training_labels = {n : labels[n] for n in training_nodes if n in labels}
            test_nodes = nodes[x:y]

            test_labelling = run_algorithm(network, training_labels, args)
            accuracy = score_cv(test_nodes, test_labelling, labels)
            accuracies.append(accuracy)

        print(f"Average Accuracy: {np.mean(accuracies)}")
        print("Cross Validation Results")
        print("========================")
        # for i in range(len(accuracies)):
        #     acc = accuracies[i]
        #     print("Fold " + str(i) + " Accuracy: " + str(acc))
    else:
        labelling = run_algorithm(network, labels, args)

        """ Print labelling to stdout """
        for node, labels_list in labelling.items():
            labels_str = " ".join(labels_list)
            print(node + " " + labels_str)

    exit(0)
