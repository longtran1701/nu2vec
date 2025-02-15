import numpy as np
import networkx as nx
import random
from tqdm import tqdm
from itertools import permutations


class Graph():
	def __init__(self, nx_G, is_directed, p, q, r, nn=None, rs=None):
		random.seed(73)
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.nn = nn
		
		if not self.nn:
			raise Exception("Must name at least 1 network")
		
		self.rs = {}
		if rs:
			if len(rs) != len(self.nn):
				raise Exception('Number of r params need to equal number of '
								+ 'networks to keep if more than 1 is passed')
			
			for i in range(len(rs)):
				self.rs[nn[i]] = rs[i]
		else:
			self.rs = {network : r for network in nn}


	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]

			cur_prefix, cur_suffix = tuple(cur.split('_'))
			other_networks = [n for n in self.nn if n != cur_suffix]
			cur_nbrs = sorted(G.neighbors(cur))
			cur_size_prev = len(cur_nbrs)
			for network in other_networks:
				try:
					cur_nbrs += sorted(G.neighbors(f'{cur_prefix}_{network}'))
				except:
					pass

			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					draw = alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])
					next_node = cur_nbrs[draw]
					walk.append(next_node)
			else:
				break

		# remove suffix so we're only left with 1 network
		return list(map(lambda x: x.split('_')[0], walk))


	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter + 1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks


	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G  = self.G
		p  = self.p
		q  = self.q
		rs = self.rs

		unnormalized_probs = []

		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)

		dst_prefix, dst_suffix = tuple(dst.split('_'))
		nn = [n for n in self.nn if n != dst_suffix]
		for network in nn:
			try:
				dst_network = f'{dst_prefix}_{network}'
				for dst_nbr in sorted(G.neighbors(dst_network)):
					unnormalized_probs.append(G[dst_network][dst_nbr]['weight'] / (rs[network] * (len(self.nn) - 1)))
			except:
				continue
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		return alias_setup(normalized_probs)


	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}

		print('------------------Processing nodes------------------')
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		
		alias_edges = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			print('------------------Processing edges------------------')
			for u, v in G.edges():
				u_v = self.get_alias_edge(u, v)
				v_u = self.get_alias_edge(v, u)
				alias_edges[(u, v)] = u_v
				alias_edges[(v, u)] = v_u
			
			print('----------------Processing networks-----------------')

			edge_set = set(G.edges())
			for u, v in permutations(G.nodes(), 2):
				u_pref, u_suf = tuple(u.split('_'))
				v_pref, v_suf = tuple(v.split('_'))
				other_networks = [n for n in self.nn if n != u_suf]
				for net in other_networks:
					u_prime = f'{u_pref}_{net}'
					v_prime = f'{v_pref}_{net}'
					if (u_prime, v_prime) in edge_set or (v_prime, u_prime) in edge_set:
						alias_edges[(u, v_prime)] = alias_edges[(u_prime, v_prime)]

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		return
	

	def print_cross_edges(self):
		for u, v in self.alias_edges.keys():
			_, u_pref = tuple(u.split('_'))
			_, v_pref = tuple(v.split('_'))
			if u_pref != v_pref:
				print(f'({u}, {v}) : {self.alias_edges[(u, v)]}')


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)
	
	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)
	
	return J, q


def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]
