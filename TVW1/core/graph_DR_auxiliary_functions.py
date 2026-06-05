"""Graph-DR topology helpers.

Builds the (Z, parent_node, d) triple consumed by `graph_DR`, and enumerates
connected base graphs for the algebraic-connectivity (lambda_1) study.
(Previously these helpers were copy-pasted inside the experiment scripts.)

Conventions (Bredies, Chenchene & Naldi 2022):
  - state graph: a complete DAG on N nodes drives the resolvent coupling;
  - base graph: a connected subgraph with N-1 columns, encoded by Z with ZZ^T = L.
"""
import numpy as np
from itertools import combinations


def complete_state_graph(N):
    """Complete DAG on N nodes: returns (E_state, parent_node, degrees d)."""
    E_state = [(i, j) for i in range(N) for j in range(i + 1, N)]
    parent_node = [[] for _ in range(N)]
    for h, i in E_state:
        parent_node[i].append(h)
    adj = [set() for _ in range(N)]
    for h, i in E_state:
        adj[h].add(i)
        adj[i].add(h)
    d = np.array([len(a) for a in adj], dtype=float)
    return E_state, parent_node, d


def incidence_Z(N, E_base):
    """Incidence matrix Z (N x len(E_base)): edge (u,v) -> column with -1 at u, +1 at v."""
    Z = np.zeros((N, len(E_base)))
    for j, (u, v) in enumerate(E_base):
        Z[u, j] = -1
        Z[v, j] = +1
    return Z


def make_graph_DR_parameters(N, E_base, E_state=None):
    """Return (Z, parent_node, d) for graph_DR; complete state graph by default."""
    if E_state is None:
        _, parent_node, d = complete_state_graph(N)
    else:
        parent_node = [[] for _ in range(N)]
        for h, i in E_state:
            parent_node[i].append(h)
        adj = [set() for _ in range(N)]
        for h, i in E_state:
            adj[h].add(i)
            adj[i].add(h)
        d = np.array([len(a) for a in adj], dtype=float)
    Z = incidence_Z(N, E_base)
    return Z, parent_node, d


# ---- Topology enumeration (lambda_1 study) ----
def is_connected(N, edges):
    if not edges:
        return N <= 1
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj[node] - visited)
    return len(visited) == N


def graph_laplacian(N, edges):
    """Laplacian of the undirected graph (each edge (u,v) listed once)."""
    L = np.zeros((N, N))
    for u, v in edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1
    return L


def onto_decomposition(L, N):
    """Z such that Z Z^T = L, shape N x (N-1)."""
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvecs[:, 1:] * np.sqrt(np.maximum(eigvals[1:], 0))


def enumerate_connected_base_graphs(N):
    """All connected subgraphs of K_N as dicts: {edges, n_edges, L, lambda1, Z}."""
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    graphs = []
    for n_edges in range(N - 1, len(all_edges) + 1):
        for edge_subset in combinations(all_edges, n_edges):
            if is_connected(N, edge_subset):
                L = graph_laplacian(N, edge_subset)
                eigvals = np.sort(np.linalg.eigvalsh(L))
                graphs.append({
                    'edges': edge_subset,
                    'n_edges': n_edges,
                    'L': L,
                    'lambda1': round(eigvals[1], 4),
                    'Z': onto_decomposition(L, N),
                })
    return graphs
