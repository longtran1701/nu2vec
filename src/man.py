import networkx as nx
import sys
import argparse


parser = argparse.ArgumentParser(description="Run PPMan.")
parser.add_argument('--keep', nargs='*', help="Networks to keep from file")
parser.add_argument('--input', nargs='?', help="Input File of Graph")
parser.add_argument('--output', nargs='?', help="Output Embedding")

Gprime = None

def keep_networks(filename, to_keep=None, directed=False):
    G = nx.Graph() if not directed else nx.DiGraph()
    
    with open(filename, 'r') as f:
        hdr = f.readline().split()

        cols_to_keep = list(range(2, len(hdr)))
        if to_keep:
            cols_to_keep = list(hdr.index(col) for col in to_keep)

        for line in f:
            data = line.split()

            G.add_nodes_from([data[0], data[1]])
            for col in cols_to_keep:
                if int(data[col]) != 0:
                    G.add_edge(
                        '{}_{}'.format(data[0], hdr[col]),
                        '{}_{}'.format(data[1], hdr[col]),
                        weight=int(data[col])
                    )
    
    with open(filename + '_output', 'w') as f2:
        for u, v in G.edges():
            f2.write('{} {} {}\n'.format(u, v, G[u][v]['weight']))
    return G


def main(argv):
    global Gprime
    G = None
    if argv.keep:
        G = keep_networks(argv.input, argv.keep)


if __name__ == '__main__':
    main(parser.parse_args())