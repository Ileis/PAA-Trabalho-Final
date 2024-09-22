from math import * # type: ignore
from random import * # type: ignore
import networkx as nx
import matplotlib.pyplot as plt
from typing import * # type: ignore
from collections import deque

from sympy import degree

T = TypeVar('T')
Node = T
Graph = Dict[Node, Set[Node]]
Edge = Tuple[Node, Node]
Segment = Tuple[Graph, Set[Node]]

def add_edge(g: Graph, e: Edge) -> None:
    g.setdefault(e[0], set()).add(e[1])
    g.setdefault(e[1], set()).add(e[0])

def add_vertex(g: Graph, v: Node) -> None:
    g.setdefault(v, set())

def get_segment(s: Segment) -> Graph:
    return s[0]

def get_attachment(s: Segment) -> Set[Node]:
    return s[1]

def is_adjacent(c: Graph, u: Node, v: Node) -> bool:
    if c[v] is None or c[u] is None: return False
    return u in c[v] or v in c[u]

def graph_to_edge_set(g: Graph) -> Set[Edge]:
    edge_set: Set[Edge] = set()

    for u in g.keys():
        for v in g[u]:
            edge_set.add((u, v))

    return edge_set

def qtd_edges(g: Graph) -> int:
    s: int = 0

    for v in g:
        s += len(v)

    return s

def degree_less_than(g: Graph, d: int) -> bool:
    for v in g:
        if len(g[v]) >= d:
            return False

    return True

#AUSLANDER-PARTER
#-----------------------------------------------------------------------------------------------------------------------
# Recebe um componente biconexo B e um ciclo simples C de B
def find_segments(g: Graph, c: Graph) -> List[Segment]:

    def dfs_segments(u: Node, p: Node) -> None:
        nonlocal g, c, visited, segments

        visited[u] = True

        if u in c:
            if u != p:
                if u not in get_segment(segments[-1]):
                    add_vertex(get_segment(segments[-1]), u)
                get_attachment(segments[-1]).add(u)
                add_edge(get_segment(segments[-1]), (p, u))
            return
        
        add_vertex(get_segment(segments[-1]), u)

        if u != p:
            add_edge(get_segment(segments[-1]), (p, u))
        for v in g[u]:
            if not visited[v] or v in c:
                dfs_segments(v, u)
            elif v not in get_segment(segments[-1])[u]:
                add_edge(get_segment(segments[-1]), (u, v))
    
    def find_chords() -> None:
        nonlocal g, c, segments

        for (u, v) in graph_to_edge_set(g):
            if u in c and v in c:
                if not is_adjacent(c, u, v):
                    segments.append((dict(), set()))
                    add_edge(get_segment(segments[-1]), (u, v))
                    get_attachment(segments[-1]).add(u)
                    get_attachment(segments[-1]).add(v)

    visited: Dict[Node, bool] = {v: False for v in g.keys()}
    segments: List[Segment] = list()

    for u in g:
        if not visited[u]:
            segments.append((dict(), set()))
            dfs_segments(u, u)
            if len(get_segment(segments[-1])) == 0:
                segments.pop()

    find_chords()

    return segments

# Retorna um ciclo simples direcionado
def edges_to_cycle(e: List[Edge]) -> Graph:
    cycle: Graph = dict()

    for edge in e:
        cycle.setdefault(edge[0], set()).add(edge[1])

    return cycle

# Retorna a sequencia de vertices percorridos pelo ciclo
def cycle_to_vertices(c: Graph) -> List[Node]:
    cycle: List[Node] = list()

    if len(c) == 0: return cycle

    vertex: Node = choice(list(c.keys()))

    for i in range(len(c)):
        cycle.append(vertex)
        vertex = list(c[vertex])[0]

    return cycle

# Retorna a lista de arestas de um ciclo
def get_edge_cycle(g: Graph) -> List[Edge]:
    def dfs_cycle(u: Node, p: Node) -> List[Edge]:
        nonlocal g, visited

        visited[u] = True

        for v in g[u]:
            if not visited[v]:
                if (rec := dfs_cycle(v, u)) != []:
                   return rec + [(v, u)] if rec[0][0] != rec[-1][1] else rec

            elif v != p:
                    return [(v, u)]
            
        return []

    cycle: List[Edge] = list()
    visited: Dict[Node, bool] = {v: False for v in g.keys()}

    root = choice(list(g.keys()))
    cycle = dfs_cycle(root, root)

    return cycle

def find_cycle(g: Graph) -> Graph:
    return edges_to_cycle(get_edge_cycle(g))

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

def label_cycle(c: List[Node], s: Segment) -> Dict[Node, int]:
    label_s: Dict[Node, int] = dict()

    i: int = 0
    for v in c:
        if v in get_attachment(s):
            label_s[v] = 2 * (i)
            i += 1
        else:
            label_s[v] = (2 * i) + 1

    return label_s

def check_conflict(seg: Segment, cycle_label_seg: Dict[Node, int]) -> bool:
    k: int = len(get_attachment(seg))
    labels: List[int] = [0] * k * 2
    s: int = 0

    for att in get_attachment(seg):
        labels[cycle_label_seg[att]] = 1
    
    for i in range(len(labels)):
        s = s + labels[i]

    part_sum: int = labels[0] + labels[1] + labels[2]

    for i in range((2 * k) - 2):
        if part_sum == s:
            return False
    
        part_sum = part_sum + labels[(3 + i) % (2 * k)] + labels[(4 + i) % (2 * k)]
        part_sum = part_sum - labels[i] - labels[(1 + i) % (2 * k)]

    return True

def get_interlacement_graph(seg: List[Segment], cycle: Graph) -> Graph:
    interlacement_graph: Graph = {f's{i}': set() for i in range(len(seg))}
    cycle_vertices: List[Node] = cycle_to_vertices(cycle)

    for i in range(len(seg)):
        for j in range(i + 1, len(seg)):
            if (check_conflict(seg[i], label_cycle(cycle_vertices, seg[j]))):
                add_edge(interlacement_graph, (f's{i}', f's{j}'))

    return interlacement_graph

def test_bipartite(g: Graph) -> bool:
    def bfs_bipartite(u) -> bool:
        nonlocal g, labels
        queue: Deque[Node] = deque()

        queue.append(u)

        while len(queue) > 0:
            v = queue.popleft()
            for w in g[v]:
                if labels[w] == -1:
                    labels[w] = 1 - labels[v]
                    queue.append(u)
                elif labels[w] == labels[v]:
                    return False
        return True

    labels: Dict[Node, int] = {v: -1 for v in g.keys()}

    for u in g.keys():
        if labels[u] == -1:
            if not bfs_bipartite(u):
                return False
            
    return True

#TODO: implementar get_sub_cycle, que retorna um ciclo que passa pelo segmento s
def get_sub_cycle(c: Graph, s: Graph) -> Graph:
    return dict()

def auslander_parter(g: Graph) -> bool:
    def _auslander_parter(b: Graph, c: Graph) -> bool:

        if len(c) == 0:
            return True
        
        segments = find_segments(b, c)

        if len(segments) == 0:
            return True
        
        if len(segments) == 1 and degree_less_than(get_segment(segments[-1]), 3):
            return True
        
        interlacement_g = get_interlacement_graph(segments, c)

        if test_bipartite(interlacement_g) == False:
            return False
    
        for seg in segments:
            sub_bi = b | get_segment(seg)

            if qtd_edges(sub_bi) > 3 * len(sub_bi) - 6 and len(sub_bi) > 2:
                return False



        return True

    bi_comp, _ = find_bi_comp(g)

    for b in bi_comp:
        if qtd_edges(b) > 3 * len(b) - 6 and len(b) > 2:
            return False
        c = find_cycle(b)

        if _auslander_parter(b, c) == False:
            return False
    
    return True
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
    bound: float = floor(((3 * v) - 6)) / 100
    return nx.erdos_renyi_graph(v, uniform(0.15, bound))

def show_graph(g: nx.Graph, color: List[str] = []) -> None:
    if color != []:
        nx.draw(g, nx.spring_layout(g), node_color=color)
    else:
        nx.draw(g, nx.spring_layout(g))

    plt.show()

def show_bipartite_graph(g: Graph) -> None:
    g_nx = my_Graph_to_nx_Graph(g)

    if nx.is_bipartite(g_nx):
        set_a, set_b = nx.bipartite.sets(g_nx)
        pos = nx.bipartite_layout(g_nx, set_a)

        nx.draw(g_nx, pos, with_labels=True)
    else:
        nx.draw(g_nx, with_labels=True)
    
    plt.show()

def show_biconnected_components(g: Graph, cutvertices: List[Node], bi_comp: List[Graph], flag_label: bool) -> None:
    g_nx = my_Graph_to_nx_Graph(g)
    bi_comp_nx = [my_Graph_to_nx_Graph(b) for b in bi_comp]

    fig, axes = plt.subplots(1, len(bi_comp_nx) + 1, figsize=(15, 5))

    nx.draw(g_nx, nx.spring_layout(g_nx), ax=axes[0], node_color=['red' if v in cutvertices else 'lightblue' for v in g_nx.nodes()], with_labels=flag_label)
    axes[0].set_title('graph')

    for i, b in enumerate(bi_comp_nx):
        nx.draw(b, nx.spring_layout(b), ax=axes[i + 1], with_labels=flag_label)
        axes[i + 1].set_title(f'graph {i + 1}')

    plt.tight_layout()
    plt.show()

def show_bi_comp_cycle_and_seg(g: Graph, c: Graph, s: List[Segment], flag_label: bool) -> None:

    # print('graph: ', get_segment(s[0]))
    # print('att:', get_attachment(s[0]))

    g_nx = my_Graph_to_nx_Graph(g)
    cycle = cycle_to_vertices(c)

    fig, axes = plt.subplots(1, len(s) + 1, figsize=(15, 5))

    pos = nx.spring_layout(g_nx, seed=42)

    circle_pos = nx.circular_layout(cycle) # type: ignore
    for node in cycle:
        pos[node] = circle_pos[node] # type: ignore

    nx.draw(g_nx, pos, ax=axes[0], node_color=['yellow' if v in cycle else 'lightblue' for v in g_nx.nodes()], with_labels=flag_label)

    for i, s_b in enumerate(s):
        nx.draw(seg := my_Graph_to_nx_Graph(get_segment(s_b)), ax=axes[i + 1], node_color=['orange' if v in cycle else 'lightblue' for v in seg.nodes()], with_labels=flag_label)

    plt.show()

def main() -> None:
    g_nx = get_possible_planar_graph(20)
    g = nx_Graph_to_my_Graph(g_nx)
    bi_comp, cutvertices = find_bi_comp(g)
    show_biconnected_components(g, cutvertices, bi_comp, True)

    bi_comp = list(filter(lambda x : len(x) > 2, bi_comp))

    for b in bi_comp:
        c = find_cycle(b)
        s = find_segments(b, c)
        show_bi_comp_cycle_and_seg(b, c, s, True)
        cycle = cycle_to_vertices(c)
        for seg in s:
            print(label_cycle(cycle, seg))
        
        show_bipartite_graph(get_interlacement_graph(s, c))
    
    pass


if __name__ == '__main__':
    main()