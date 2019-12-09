import networkx as nx
import sys
import argparse
import time
import subprocess
from os import remove


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPMan.")
    parser.add_argument('--keep', nargs='*', help="Networks to keep from file", default=None)
    parser.add_argument('--input', nargs='?', help="Input File of Graph")
    parser.add_argument('--output', nargs='?', help="Output Embedding")
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--r', type=float, default=1,
                        help='Teleport hyperparameter. Default is 1.')
    parser.add_argument('--rs', nargs='*', default=None, 
                        help="Teleportation parameters into networks")
    return parser.parse_args()


def keep_networks(filename, to_keep, directed=False):
    G = nx.Graph() if not directed else nx.DiGraph()
    
    with open(filename, 'r') as f:
        hdr = f.readline().split()
        cols_to_keep = list(hdr.index(col) for col in to_keep)

        for line in f:
            data = line.split()

            for col in cols_to_keep:
                if int(data[col]) != 0:
                    G.add_edge(
                        '{}_{}'.format(data[0], hdr[col]),
                        '{}_{}'.format(data[1], hdr[col]),
                        weight=float(data[col])
                    )
    
    return G


def normalize_edges_by_component(G, to_keep):
    weight_map = {component : {} for component in to_keep}
    for u, v in G.edges():
        _, comp = tuple(u.split('_'))
        weight_map[comp][(u, v)] = float(G[u][v]['weight'])
    
    for comp, component_edges in weight_map.items():
        norm_const = sum(component_edges.values())
        for u, v in component_edges.keys():
            weight_map[comp][(u, v)] /= float(norm_const)
            G[u][v]['weight'] = weight_map[comp][(u, v)]
    
    return G


def run_node2vec(G, args):
    filename = f'{args.input}.{".".join(args.keep)}.p.{args.p}.q.{args.q}.'
    if not args.rs:
        filename += f'r.{args.r}.tmp'
    else:
        for i in range(len(args.keep)):
            filename += f'r_{args.keep[i]}.{args.rs[i]}.'
        filename += 'tmp'

    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write('{} {} {}\n'.format(u, v, G[u][v]['weight']))

    r_to_use = ['--r', str(args.r)] if not args.rs else ['--rs'] + [str(r) for r in args.rs]    
    n2v = ['python3', 'main.py', '--input', filename, '--output', filename + '.emb',
           '--p', str(args.p), '--q', str(args.q), '--nn'] + args.keep + ['--weighted',
           '--undirected'] + r_to_use

    subprocess.call(n2v)
    
    remove(filename)


def main(argv):
    if argv.rs is not None and len(argv.rs) != len(argv.keep):
            raise Exception('Number of r params need to equal number of networks to keep')
        
    G = keep_networks(argv.input, to_keep=argv.keep)
    G = normalize_edges_by_component(G, to_keep=argv.keep)
    run_node2vec(G, argv)


if __name__ == '__main__':
    args = parse_args()
    main(args)