from typing import List, Tuple, Dict, Set

Node = int
Graph = Dict[Node, Set[Node]]
Edge = Tuple[Node, Node]
RootedTree = Tuple[Node, Graph]

def main() -> None:
    g: Graph = graph(list(range(8)))

    add_edge(g, 0, 1)
    add_edge(g, 0, 2)
    add_edge(g, 1, 3)
    add_edge(g, 1, 4)
    add_edge(g, 2, 5)
    add_edge(g, 3, 6)
    add_edge(g, 4, 5)
    add_edge(g, 4, 6)
    add_edge(g, 5, 7)
    add_edge(g, 6, 7)

    h = depth_first_search(g, 0)

    for i in h:
        print(i)

# def contraction(post_order: List[Node], it: int, dfs_tree: RootedTree, g_back: Graph) -> None:
#     candidate: List[Node] = list()

#     for i in range(it):
#         if post_order[i] in g_back[it]:
#             candidate.append(i)


def graph(v: List[Node]) -> Graph:
    return {x: set() for x in v}

def add_edge(g: Graph, v: Node, u: Node) -> bool:
    if g.get(v) is not None and g.get(u) is not None:
        g[v].add(u)
        g[u].add(v)
        return True
    return False

def set_edge_to_graph(set_edge: Set[Edge]) -> Graph:
    g: Graph = {}
    for e in set_edge:
        g.setdefault(e[0], set()).add(e[1])
        g.setdefault(e[1], set()).add(e[0])
    return g

def depth_first_search(g: Graph, s: Node) -> Tuple[List[Node], Graph, Graph]:
    tree_edge: Set[Edge] = set()
    back_edge: Set[Edge] = set()
    visited: List[bool] = [False] * len(g)
    finish: List[Node] = list()

    _depth_first_search(g, s, visited, tree_edge, back_edge, finish)

    depth_tree: Graph = set_edge_to_graph(tree_edge)
    g_back_edge: Graph = set_edge_to_graph(back_edge)

    return (finish, depth_tree, g_back_edge)

def _depth_first_search(g: Graph, s: Node, visited: List[bool], tree_edge: Set[Edge], back_edge: Set[Edge], finish: List[Node]) -> None:
    visited[s] = True

    for v in g[s]:
        if not visited[v]:
            tree_edge.add((s, v))
            _depth_first_search(g, v, visited, tree_edge, back_edge, finish)
        else:
            back_edge.add((s, v))

    finish.append(s)
    
if __name__ == '__main__':
    main()