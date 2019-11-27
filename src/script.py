from man import keep_networks, normalize_edges_by_component
import matplotlib.pyplot as plt
import networkx as nx


G = keep_networks('small_net.txt', to_keep=['n1', 'n2'])
G = normalize_edges_by_component(G, to_keep=['n1', 'n2'])
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, font_size=10, edge_labels=labels)
plt.show()