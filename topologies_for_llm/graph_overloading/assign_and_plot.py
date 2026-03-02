import random
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os

from collections import defaultdict
import networkx as nx


def assign_od_to_edges_shortest(
    G: nx.DiGraph,
    OD_csr,
    *,
    weight: str = "weight",
    split_equal_shortest: bool = False,
):
    """
    Route each nonzero OD demand on shortest paths and accumulate per-edge load.

    If split_equal_shortest=False (default):
        - Uses ONE shortest path per (s,t) (NetworkX tie-break / adjacency order).
    If split_equal_shortest=True:
        - Splits demand evenly across ALL equal-cost shortest paths (ECMP-style).

    Returns: dict[(u,v)] -> load
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

            if not split_equal_shortest:
                path = nx.shortest_path(G, source=s, target=t, weight=weight)
                for u, v in zip(path[:-1], path[1:]):
                    edge_load[(u, v)] += demand
            else:
                paths = list(nx.all_shortest_paths(G, source=s, target=t, weight=weight))
                k = len(paths)
                if k == 0:
                    continue
                share = demand / k
                for path in paths:
                    for u, v in zip(path[:-1], path[1:]):
                        edge_load[(u, v)] += share

    return dict(edge_load)



def annotate_graph_with_loads(G, edge_load, *, capacity=None):
    """
    capacity=None means "infinite" (no overload by util). We store:
      - load
      - util (None if capacity is None)
      - overloaded (always False if capacity is None)
    """
    for u, v, data in G.edges(data=True):
        load = float(edge_load.get((u, v), 0.0))
        data["load"] = load

        if capacity is None:
            data["util"] = None
            data["overloaded"] = False
            data["capacity"] = math.inf
        else:
            cap = float(capacity)
            data["capacity"] = cap
            data["util"] = load / cap if cap > 0 else 0.0
            data["overloaded"] = data["util"] > 1.0


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

def draw_fattree_load_heat(
    G: nx.DiGraph,
    *,
    max_edges_to_draw: int = 3000,
    title: str = "FatTree edge load (heat)",
    cmap: str = "inferno",
    use_log_norm: bool = True,
    min_positive: float = 1e-12,
):
    all_edges = list(G.edges())
    all_loads = np.array([G[u][v].get("load", 0.0) for u, v in all_edges], dtype=float)

    if not all_edges:
        print("Graph has no edges to draw.")
        return

    # Keep only top-K loaded edges
    if max_edges_to_draw is not None and len(all_edges) > max_edges_to_draw:
        idx = np.argsort(all_loads)[::-1][:max_edges_to_draw]
        edges = [all_edges[i] for i in idx]
        loads = all_loads[idx]
    else:
        edges = all_edges
        loads = all_loads

    max_load = float(np.max(loads)) if loads.size else 1.0
    if max_load <= 0:
        max_load = 1.0

    # Normalization
    if use_log_norm:
        positive = loads[loads > 0]
        vmin = float(np.min(positive)) if positive.size else min_positive
        vmin = max(vmin, min_positive)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=max_load)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=max_load)

    cm = mpl.cm.get_cmap(cmap)

    # Colors + widths
    edge_colors = [cm(norm(max(l, min_positive if use_log_norm else 0.0))) for l in loads]
    if use_log_norm:
        wscale = np.array([norm(max(l, min_positive)) for l in loads], dtype=float)
    else:
        wscale = np.array([l / max_load for l in loads], dtype=float)
    edge_widths = (1.0 + 5.0 * wscale).tolist()

    pos = nx.spring_layout(G, seed=0, k=0.25)

    # ✅ Explicit fig/ax so colorbar knows where to go
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=30, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=False,
        alpha=0.95,
        ax=ax,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Edge load (bytes in matrix window)" + (" [log scale]" if use_log_norm else ""))

    ax.set_title(f"{title} (max load {float(np.max(loads)):.3e})")
    ax.set_axis_off()
    plt.show()

def plot_edge_load_cdf(
    G,
    *,
    title: str = "Edge-load CDF",
    use_log_x: bool = True,
    include_zeros: bool = True,
):
    """
    Plots: percent of edges with load <= x  (empirical CDF)

    Requires each edge to have attribute 'load' (float).
    """
    loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)

    if not include_zeros:
        loads = loads[loads > 0]

    if loads.size == 0:
        print("No loads to plot (empty or all zero).")
        return

    loads.sort()
    y = (np.arange(1, loads.size + 1) / loads.size) * 100.0  # percent

    plt.figure(figsize=(10, 6))
    plt.plot(loads, y)
    plt.ylabel("Edges with load ≤ x (%)")
    plt.xlabel("Edge load (bytes in matrix window)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    if use_log_x:
        # log scale helps when loads span orders of magnitude
        plt.xscale("log")

    plt.ylim(0, 105)
    plt.show()


def plot_edge_load_cdf_multiple(
    graphs_dict,
    *,
    title: str = "Edge-load CDF (Multiple Graphs)",
    use_log_x: bool = True,
    include_zeros: bool = True,
    save_dir: str | None = None,
    filename: str | None = None,
):
    """
    Plots: percent of edges with load <= x  (empirical CDF) for multiple graphs.

    Args:
        graphs_dict: Dictionary mapping string labels to NetworkX graphs. Each edge must have attribute 'load' (float).
        title: Plot title.
        use_log_x: Whether to use log scale on x-axis.
        include_zeros: Whether to include zero loads in the calculation.
    """
    if not graphs_dict:
        print("No graphs provided.")
        return

    plt.figure(figsize=(10, 6))
    
    # Use a colormap to generate distinct colors for each graph
    num_graphs = len(graphs_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, num_graphs)) if num_graphs <= 10 else plt.cm.tab20(np.linspace(0, 1, num_graphs))
    
    for idx, (label, G) in enumerate(graphs_dict.items()):
        loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)

        if not include_zeros:
            loads = loads[loads > 0]

        if loads.size == 0:
            print(f"Graph '{label}' has no loads to plot (empty or all zero).")
            continue

        loads.sort()
        y = (np.arange(1, loads.size + 1) / loads.size) * 100.0  # percent
        
        plt.plot(loads, y, color=colors[idx], label=label, linewidth=2)

    plt.ylabel("Edges with load ≤ x (%)")
    plt.xlabel("Edge load (bytes in matrix window)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    if use_log_x:
        # log scale helps when loads span orders of magnitude
        plt.xscale("log")

    plt.ylim(0, 105)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else f"{title}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    