from man import keep_networks
import matplotlib.pyplot as plt
import networkx as nx


G = keep_networks('small_net.txt', to_keep=['n1', 'n2'])
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
plt.show()