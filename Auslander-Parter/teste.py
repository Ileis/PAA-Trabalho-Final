import matplotlib.pyplot as plt
import networkx as nx

# Criação do grafo
G = nx.Graph()

# Adiciona arestas ao grafo
edges = [(1, 2), (2, 3), (3, 4), (4, 1),  # Ciclo (1-2-3-4-1)
         (1, 5), (2, 6), (3, 7), (4, 8)]  # Arestas externas

G.add_edges_from(edges)

# Definindo o ciclo que queremos destacar
cycle = [1, 2, 3, 4]

# Obter as posições de todos os nós
pos = nx.spring_layout(G, seed=42)  # Layout padrão com spring

# Ajustar manualmente a posição dos nós do ciclo para ficarem em um círculo
circle_pos = nx.circular_layout(cycle)  # Posição circular para os nós do ciclo
for node in cycle:
    pos[node] = circle_pos[node]  # Reposiciona os nós do ciclo

# Desenhar o grafo completo
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')

# Destacar o ciclo específico
nx.draw_networkx_edges(G, pos, edgelist=[(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))],
                       edge_color='red', width=2.5)  # Ciclo em vermelho

# Desenhar o restante das arestas (excluindo as do ciclo)
other_edges = [edge for edge in G.edges() if edge not in [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]]
nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='gray', style='dashed')

# Mostrar o gráfico
plt.show()
