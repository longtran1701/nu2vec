import argparse

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

def knn(embedding, labels, k):
    pass

if __name__ == "__main__":
    args = parse_args()

    labels = parse_labels(args.labels)
    print(labels)

    if args.mv:
        print("Majority Vote")
    elif args.wmv:
        print("Weighted Majority Vote")
    elif args.knn is not None:
        print("KNN: " + str(args.knn))

    exit(0)
