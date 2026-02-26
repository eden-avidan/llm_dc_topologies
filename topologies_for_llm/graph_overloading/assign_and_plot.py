import random
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def assign_od_to_edges_shortest_first(G: nx.DiGraph, OD_csr, weight="weight"):
    """
    Routing model #1:
      shortest path; if multiple shortest paths exist, NetworkX returns one
      (your FatTree.convert_to_networkx sorts edges, so tie-break is stable/"first by order").
    """
    edge_load = defaultdict(float)

    n = OD_csr.shape[0]
    for s in range(n):
        row_start = OD_csr.indptr[s]
        row_end = OD_csr.indptr[s + 1]
        if row_start == row_end:
            continue

        js = OD_csr.indices[row_start:row_end]
        ds = OD_csr.data[row_start:row_end]

        for t, demand in zip(js, ds):
            t = int(t)
            demand = float(demand)
            if demand <= 0 or s == t:
                continue

            path = nx.shortest_path(G, source=s, target=t, weight=weight)
            for u, v in zip(path[:-1], path[1:]):
                edge_load[(u, v)] += demand

    return edge_load


def annotate_graph_with_loads(G: nx.DiGraph, edge_load: dict):
    """
    Writes load/util/overloaded back into edge attributes.
    """
    for u, v, data in G.edges(data=True):
        load = float(edge_load.get((u, v), 0.0))
        cap = float(data.get("capacity", 1.0))
        util = load / cap if cap > 0 else 0.0
        data["load"] = load
        data["util"] = util
        data["overloaded"] = util > 1.0


def draw_fattree_overload(G: nx.DiGraph, *, max_edges_to_draw=None, title="FatTree edge utilization"):
    """
    Draw a simplified visualization:
    - edge width proportional to utilization
    - overloaded edges colored red
    Note: large graphs will be hard to visualize; consider filtering.
    """
    # Optional: filter down to "busy" edges for visibility
    edges = list(G.edges())
    if max_edges_to_draw is not None and len(edges) > max_edges_to_draw:
        edges = sorted(edges, key=lambda e: G[e[0]][e[1]].get("util", 0.0), reverse=True)[:max_edges_to_draw]

    # widths/colors from util
    widths = []
    colors = []
    for u, v in edges:
        util = G[u][v].get("util", 0.0)
        widths.append(1 + 6 * min(util, 2.0))  # cap visual thickness at util=2
        colors.append("red" if util > 1.0 else "gray")

    # a decent default layout for layered-ish graphs:
    pos = nx.spring_layout(G, seed=0, k=0.25)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=30)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color=colors, arrows=False)
    plt.title(title)
    plt.axis("off")
    plt.show()