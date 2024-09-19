from math import floor
from random import choice
import networkx as nx
import matplotlib.pyplot as plt
from typing import * # type: ignore

T = TypeVar('T')
Node = T
Graph = Dict[Node, Set[Node]]
Edge = Tuple[Node, Node]

def add_edge(g: Graph, e: Edge) -> None:
    g.setdefault(e[0], set()).add(e[1])
    g.setdefault(e[1], set()).add(e[0])

#AUSLANDER-PARTER
#-----------------------------------------------------------------------------------------------------------------------


# Retorna um ciclo simples de G
def get_cycle(g: Graph) -> List[Edge]:
    def dfs_cycle(u: Node, p: Node) -> List[Edge]:
        nonlocal g, visited

        visited[u] = True

        for v in g[u]:
            if not visited[v]:
                if (rec := dfs_cycle(v, u)) != []:
                    if rec[0][0] != rec[-1][1]:
                        return rec + [(v, u)]
                    else:
                        return rec

            elif v != p:
                    return [(v, u)]
            
        return []

    cycle: List[Edge] = list()
    visited: Dict[Node, bool] = {v: False for v in g.keys()}

    root = choice(list(g.keys()))
    cycle = dfs_cycle(root, root)

    return cycle

# Retorna uma lista de grafos (componentes biconexos) e uma lista de vertices (articulacoes).
def find_bi_comp(g: Graph) -> Tuple[List[Graph], List[Node]]:
    def dfs_bi_comp(u: Node, p: Node) -> None:
        nonlocal g, visited, children, edges_bi_comp, bi_comp, discover, low, cut_vertices, time

        visited[u] = True
        low[u] = discover[u] = time
        children[u] = 0
        time += 1

        for v in g[u]:
            if v == p: continue
            if not visited[v]:
                children[u] += 1
                edges_bi_comp.append((u, v))
                dfs_bi_comp(v, u)

                if low[v] >= discover[u]:

                    if p != u or children[u] > 1:
                        cut_vertices.append(u)

                    b: Graph = dict()
                    edge_b: Edge = tuple()

                    while edge_b != (u, v):
                        edge_b = edges_bi_comp.pop()
                        add_edge(b, edge_b)

                    bi_comp.append(b)

                low[u] = min(low[u], low[v])

            elif discover[v] < discover[u]:
                edges_bi_comp.append((u, v))
                low[u] = min(low[u], discover[v])

        if u == p:
            if children[u] >= 2:
                cut_vertices.append(u)
            if children[u] == 0:
                bi_comp.append({u: set()})
                        
    visited: Dict[Node, int] = {v: False for v in g.keys()}
    children: Dict[Node, int] = {v: 0 for v in g.keys()}

    edges_bi_comp: List[Edge] = list()
    bi_comp: List[Graph] = list()
    
    discover: Dict[Node, int] = dict()
    low: Dict[Node, int] = dict()

    cut_vertices: List[Node] = list()
    time: int = 0

    for u in g.keys():
        if not visited[u]:
            dfs_bi_comp(u, u)

    return (bi_comp, cut_vertices)

#-----------------------------------------------------------------------------------------------------------------------

# Converte um grafo (Dict[Node, Set[Node]]) para um nx.Graph.
def my_Graph_to_nx_Graph(g: Graph) -> nx.Graph:
    g_output: nx.Graph = nx.Graph()

    for v in g:
        g_output.add_node(v)
        for u in g[v]:
            g_output.add_edge(v, u)

    return g_output

# Converte um nx.Graph para um grafo (Dict[Node, Set[Node]]).
def nx_Graph_to_my_Graph(g: nx.Graph) -> Graph:
    g_output: Graph = dict()

    for v in g.nodes():
        g_output[v] = set()

    for e in g.edges():
        g_output[e[0]].add(e[1])
        g_output[e[1]].add(e[0])

    return g_output

# Retorna um nx.Graph conexo
def random_connected_graph(n_vertex: int, p: float) -> nx.Graph:
    g: nx.Graph = nx.erdos_renyi_graph(n_vertex, p)
    while not nx.is_connected(g):
        g = nx.erdos_renyi_graph(n_vertex, p)

    return g

# Retorna um nx.Graph que respeita o Teorema de Euler:
#       |E| <= 3|V| - 6.
# |E| = floor(3|V| - 6)
def get_possible_planar_graph(v: int) -> nx.Graph:
    return nx.erdos_renyi_graph(v, floor(((3 * v) - 6) / 100))

def show_graph(g: nx.Graph, color: List[str] = []) -> None:
    if color != []:
        nx.draw(g, nx.spring_layout(g), node_color=color)
    else:
        nx.draw(g, nx.spring_layout(g))

    plt.show()


def main() -> None:
    g_nx = random_connected_graph(20, 0.15)
    g_my = nx_Graph_to_my_Graph(g_nx)

    print(get_cycle(g_my))

    # bi_comp, cutvertice = find_bi_comp(g_my)

    # g_nx = my_Graph_to_nx_Graph(g_my)
    # bi_comp = [my_Graph_to_nx_Graph(b) for b in bi_comp]

    # fig, axes = plt.subplots(1, len(bi_comp) + 1, figsize=(15, 5))

    # nx.draw(g_nx, nx.spring_layout(g_nx), ax=axes[0], node_color=['red' if v in cutvertice else 'lightblue' for v in g_nx.nodes()])
    # axes[0].set_title('graph')

    # for i, g in enumerate(bi_comp):
    #     nx.draw(g, nx.spring_layout(g), ax=axes[i + 1])
    #     axes[i + 1].set_title(f'graph {i + 1}')

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()