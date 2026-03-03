import random
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import math
import os


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
    
    # Markers for distinguishing lines
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '<', '>', '8', 'P', 'X']
    
    # First pass: find global minimum first_positive across all graphs for common x_start
    global_min_positive = float('inf')
    for G in graphs_dict.values():
        loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)
        if not include_zeros:
            loads = loads[loads > 0]
        if loads.size == 0:
            continue
        positive_loads = loads[loads > 0]
        if positive_loads.size > 0:
            global_min_positive = min(global_min_positive, float(np.min(positive_loads)))
    
    # Calculate global x_start so all plots begin at the same point
    if global_min_positive == float('inf'):
        global_min_positive = 1.0  # fallback
    global_x_start = global_min_positive / 10 if use_log_x else max(0, global_min_positive - 1)
    
    for idx, (label, G) in enumerate(graphs_dict.items()):
        loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)

        if not include_zeros:
            loads = loads[loads > 0]

        if loads.size == 0:
            print(f"Graph '{label}' has no loads to plot (empty or all zero).")
            continue

        loads.sort()
        y = (np.arange(1, loads.size + 1) / loads.size) * 100.0  # percent
        
        # Find where first non-zero load starts
        num_zeros = np.searchsorted(loads, 0, side='right')
        first_positive = loads[num_zeros] if num_zeros < len(loads) else loads[-1]
        
        # Build the plot data: start with zeros at y=0 from global_x_start, then the actual CDF
        if num_zeros > 0:
            # There are zeros - show them as a horizontal line at y=0
            # Plot: global_x_start -> first_positive at y=0, then the CDF rises
            plot_x = np.concatenate([[global_x_start, first_positive], loads[num_zeros:]])
            plot_y = np.concatenate([[0, 0], y[num_zeros:]])
        else:
            # No zeros - draw horizontal at y=0 from global_x_start to first data point
            plot_x = np.concatenate([[global_x_start, loads[0]], loads])
            plot_y = np.concatenate([[0, 0], y])
        
        marker = markers[idx % len(markers)]
        plt.plot(plot_x, plot_y, color=colors[idx], label=label, linewidth=2, 
                 marker=marker, markersize=5, markevery=max(1, len(plot_x) // 15))

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


def plot_edge_load_percentiles_multiple(
    graphs_dict,
    *,
    percentiles: list[float] | None = None,
    title: str = "Edge-load Percentiles (Multiple Graphs)",
    use_log_y: bool = True,
    include_zeros: bool = True,
    connect_points: bool = True,
    label_points: bool = True,
    save_dir: str | None = None,
    filename: str | None = None,
):
    """
    Plots percentile (quantile) comparison across multiple graphs.
    
    X-axis: percentile (%), Y-axis: load value at that percentile.
    Each point represents the x-th percentile, meaning x% of observations are ≤ the plotted value.

    Args:
        graphs_dict: Dictionary mapping string labels to NetworkX graphs. Each edge must have attribute 'load' (float).
        percentiles: List of percentiles to compute (default: [10, 25, 50, 75, 90, 95, 99]).
        title: Plot title.
        use_log_y: Whether to use log scale on y-axis (recommended when values span orders of magnitude).
        include_zeros: Whether to include zero loads in the calculation.
        connect_points: Whether to connect percentile points with lines.
        label_points: Whether to label each point with its percentile and value.
        save_dir/filename: If set, save plot to disk. Otherwise show.
    """
    if not graphs_dict:
        print("No graphs provided.")
        return
    
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    percentiles = sorted(percentiles)
    
    plt.figure(figsize=(12, 7))
    
    # Use a colormap to generate distinct colors for each graph
    num_graphs = len(graphs_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, num_graphs)) if num_graphs <= 10 else plt.cm.tab20(np.linspace(0, 1, num_graphs))
    
    # Markers for distinguishing lines
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '<', '>', '8', 'P', 'X']
    
    # Offset for label positioning to avoid overlap
    label_offsets = [(-5, 10), (5, 10), (-5, -15), (5, -15), (0, 15), (0, -20)]
    
    for idx, (label, G) in enumerate(graphs_dict.items()):
        loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)

        if not include_zeros:
            loads = loads[loads > 0]

        if loads.size == 0:
            print(f"Graph '{label}' has no loads to plot (empty or all zero).")
            continue

        # Compute percentile values
        percentile_values = np.percentile(loads, percentiles)
        
        marker = markers[idx % len(markers)]
        color = colors[idx]
        
        if connect_points:
            plt.plot(percentiles, percentile_values, color=color, label=label, linewidth=2,
                     marker=marker, markersize=8)
        else:
            plt.scatter(percentiles, percentile_values, color=color, label=label, 
                        marker=marker, s=80)
        
        # Label each point with percentile and value
        if label_points:
            offset = label_offsets[idx % len(label_offsets)]
            for p, v in zip(percentiles, percentile_values):
                # Format value nicely
                if v == 0:
                    val_str = "0"
                elif v >= 1e6:
                    val_str = f"{v:.2e}"
                elif v >= 1000:
                    val_str = f"{v/1000:.1f}K"
                elif v >= 1:
                    val_str = f"{v:.1f}"
                else:
                    val_str = f"{v:.2e}"
                plt.annotate(f"P{int(p)}:{val_str}", (p, v), 
                             textcoords="offset points", xytext=offset,
                             fontsize=7, color=color, alpha=0.8)

    plt.xlabel("Percentile (%)")
    plt.ylabel("Edge load (bytes in matrix window)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper left')
    
    # Set x-axis ticks to the percentiles
    plt.xticks(percentiles)
    plt.xlim(min(percentiles) - 5, max(percentiles) + 5)

    if use_log_y:
        plt.yscale("log")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else f"{title}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    

def plot_edge_load_bucket_hist_multiple(
    graphs_dict,
    *,
    title: str = "Edge-load bucket comparison",
    include_zeros: bool = True,
    use_log_buckets: bool = True,
    num_buckets: int = 12,
    bucket_edges: np.ndarray | None = None,
    x_label: str = "Edge load bucket",
    y_label: str = "Edges in bucket (%)",
    save_dir: str | None = None,
    filename: str | None = None,
):
    """
    Bucketed comparison across multiple graphs.

    For each graph:
      - Collect edge loads from edge attribute 'load'
      - Bucket them into common bins
      - Plot grouped bars: x=buckets, y=percent of edges in bucket

    Args:
        graphs_dict: dict[label -> NetworkX graph]. Legend uses dict keys.
        include_zeros: If False, remove 0-load edges before bucketing.
        use_log_buckets: If True, bins are log-spaced (recommended when loads span orders of magnitude).
        num_buckets: Number of buckets if bucket_edges is not provided.
        bucket_edges: Optional explicit bin edges (length = num_buckets+1). If provided, overrides use_log_buckets/num_buckets.
        save_dir/filename: If set, save plot to disk. Otherwise show.
    """
    if not graphs_dict:
        print("No graphs provided.")
        return

    labels = list(graphs_dict.keys())

    # ---- collect loads and also global min/max (for shared bins) ----
    loads_by_label: dict[str, np.ndarray] = {}
    global_vals = []

    for label, G in graphs_dict.items():
        loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)
        if not include_zeros:
            loads = loads[loads > 0]
        loads_by_label[label] = loads

        if loads.size > 0:
            global_vals.append(loads)

    if not global_vals:
        print("All graphs have no loads to plot (empty or all zero).")
        return

    global_vals = np.concatenate(global_vals)
    global_min = float(np.min(global_vals))
    global_max = float(np.max(global_vals))

    # ---- build common bucket edges ----
    if bucket_edges is None:
        if use_log_buckets:
            # log buckets can't include 0; if zeros included, they will be handled separately via a first bucket
            # We'll make bins over positive range and optionally add a zero bucket.
            positive = global_vals[global_vals > 0]
            if positive.size == 0:
                # everything is zero
                bucket_edges = np.array([0.0, 1.0], dtype=float)
                num_buckets = 1
            else:
                pmin = float(np.min(positive))
                pmax = float(np.max(positive))
                # avoid degenerate ranges
                if pmax <= pmin:
                    pmax = pmin * 10.0
                
                # Linear stepping within each order of magnitude (e.g., 1, 2, 3, ... or 1, 1.5, 2, ...)
                # Using coefficient step of 1 (can change to 0.5 for finer granularity)
                coeff_step = 2.0  # coefficients: 1, 2, 3, ..., 9  (use 0.5 for 1, 1.5, 2, ...)
                
                exp_min = int(np.floor(np.log10(pmin)))
                exp_max = int(np.ceil(np.log10(pmax)))
                
                edges = []
                for exp in range(exp_min, exp_max + 1):
                    base = 10.0 ** exp
                    coeffs = np.arange(1.0, 10.0, coeff_step)
                    for c in coeffs:
                        val = c * base
                        if val >= pmin and val <= pmax * 1.01:  # small tolerance
                            edges.append(val)
                
                # Ensure we have the endpoints
                if not edges or edges[0] > pmin:
                    edges.insert(0, pmin)
                if edges[-1] < pmax:
                    edges.append(pmax)
                
                bucket_edges = np.array(sorted(set(edges)), dtype=float)

                # If we include zeros, prepend a [0, first_edge) bucket.
                if include_zeros:
                    bucket_edges = np.concatenate(([0.0], bucket_edges))
        else:
            # linear buckets across full range
            if global_max <= global_min:
                global_max = global_min + 1.0
            bucket_edges = np.linspace(global_min, global_max, num_buckets + 1)

    bucket_edges = np.asarray(bucket_edges, dtype=float)
    if bucket_edges.ndim != 1 or bucket_edges.size < 2:
        raise ValueError("bucket_edges must be a 1D array with at least 2 entries.")
    if not np.all(bucket_edges[1:] >= bucket_edges[:-1]):
        raise ValueError("bucket_edges must be non-decreasing.")

    # ---- compute bucket percentages for each graph ----
    # hist[i] = count in bin i where bins are [edge[i], edge[i+1])
    bucket_counts = []
    for label in labels:
        loads = loads_by_label[label]
        if loads.size == 0:
            counts = np.zeros(bucket_edges.size - 1, dtype=float)
        else:
            counts, _ = np.histogram(loads, bins=bucket_edges)
        # convert to percent
        denom = loads.size if loads.size > 0 else 1.0
        bucket_counts.append(100.0 * counts / denom)

    bucket_counts = np.stack(bucket_counts, axis=0)  # shape: (num_graphs, num_bins)

    # ---- format x tick labels ----
    def _fmt(v: float) -> str:
        if v == 0.0:
            return "0"
        if v < 1e-10:
            return "0"
        # Extract coefficient and exponent, show as "Ce±X" (e.g., "2e6", "1.5e9")
        exp = int(np.floor(np.log10(abs(v))))
        coeff = v / (10.0 ** exp)
        # Clean up coefficient display
        if abs(coeff - round(coeff)) < 0.01:
            coeff_str = str(int(round(coeff)))
        else:
            coeff_str = f"{coeff:.1f}".rstrip('0').rstrip('.')
        return f"{coeff_str}e{exp}"

    bin_labels = []
    for i in range(bucket_edges.size - 1):
        left = bucket_edges[i]
        right = bucket_edges[i + 1]
        # Show only the left edge value for cleaner labels
        bin_labels.append(_fmt(left))

    # ---- plot grouped bars ----
    num_bins = bucket_edges.size - 1
    num_graphs = len(labels)

    x = np.arange(num_bins, dtype=float)
    group_width = 0.85
    bar_width = group_width / max(1, num_graphs)

    plt.figure(figsize=(max(10, num_bins * 0.6), 6))

    for gi, label in enumerate(labels):
        offsets = x - group_width / 2 + (gi + 0.5) * bar_width
        plt.bar(offsets, bucket_counts[gi], width=bar_width, label=label)

    plt.xticks(x, bin_labels, rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else f"{title}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_shortest_path_heatmap(
    G: nx.Graph | nx.DiGraph,
    *,
    title: str = "Shortest Path Length Heatmap",
    node_order: list | None = None,
    num_endpoints: int | None = None,
    num_ticks: int = 42,
    x_label: str = "Target Node",
    y_label: str = "Source Node",
    save_dir: str | None = None,
    filename: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 300,
):
    """
    Generate a heatmap showing shortest path lengths (number of edges) between all node pairs.
    
    The heat value for cell (i, j) represents the number of edges in the shortest path
    from node i to node j. Unreachable pairs are shown as NaN (masked/white).
    
    Args:
        G: NetworkX graph (directed or undirected).
        title: Plot title.
        node_order: Optional list specifying the order of nodes in the matrix.
                    If None, uses sorted(G.nodes()) or endpoints only if num_endpoints is set.
        num_endpoints: If set, only include nodes 0 to num_endpoints-1 (GPU/endpoint nodes).
                       This is useful to filter out switch/router nodes and show only
                       GPU-to-GPU distances matching the transport matrix order.
        num_ticks: Number of ticks to show on x and y axes (default: 42).
        x_label: Label for x-axis (target nodes).
        y_label: Label for y-axis (source nodes).
        save_dir/filename: If set, save plot to disk. Otherwise show.
        figsize: Figure size as (width, height).
        dpi: Resolution for saved figure.
    """
    if G.number_of_nodes() == 0:
        print("Graph has no nodes.")
        return
    
    # Determine node order
    if node_order is None:
        if num_endpoints is not None:
            # Use only endpoint nodes (0 to num_endpoints-1), matching transport matrix order
            node_order = list(range(num_endpoints))
        else:
            try:
                node_order = sorted(G.nodes())
            except TypeError:
                # Nodes not sortable, use arbitrary order
                node_order = list(G.nodes())
    
    n = len(node_order)
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    
    # Compute all-pairs shortest path lengths
    # Using unweighted shortest path (number of edges)
    dist_matrix = np.full((n, n), np.nan, dtype=float)
    
    for source in node_order:
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            src_idx = node_to_idx[source]
            for target, length in lengths.items():
                if target in node_to_idx:
                    tgt_idx = node_to_idx[target]
                    dist_matrix[src_idx, tgt_idx] = length
        except nx.NetworkXError:
            # Source not in graph or other error
            continue
    
    # Determine discrete values present in the matrix
    valid_values = dist_matrix[~np.isnan(dist_matrix)]
    if valid_values.size == 0:
        print("No reachable paths in the graph.")
        return
    
    min_dist = int(np.min(valid_values))
    max_dist = int(np.max(valid_values))
    discrete_values = list(range(min_dist, max_dist + 1))
    
    # Create discrete colormap using viridis
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(discrete_values))))
    
    # Boundaries for discrete colors
    bounds = np.arange(len(discrete_values) + 1) - 0.5 + min_dist
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        dist_matrix,
        cmap=cmap,
        norm=norm,
        square=True,
        cbar=True,
        mask=np.isnan(dist_matrix),
        vmin=min(discrete_values),
        vmax=max(discrete_values),
        linewidths=0,
        linecolor=None,
        xticklabels=False,  # Hide individual tick labels for large graphs
        yticklabels=False,
    )
    
    # Configure colorbar with discrete ticks
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks(discrete_values)
    cbar.set_ticklabels([str(v) for v in discrete_values])
    cbar.set_label("Shortest Path Length (edges)")
    
    # Add linear ticks on x and y axes
    ax = plt.gca()
    if num_ticks > 0 and n > 1:
        # Calculate tick positions (linear spacing)
        tick_positions = np.linspace(0, n - 1, min(num_ticks, n)).astype(int)
        # Get corresponding node labels
        tick_labels = [str(node_order[i]) for i in tick_positions]
        
        ax.set_xticks(tick_positions + 0.5)  # +0.5 to center on cells
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticks(tick_positions + 0.5)
        ax.set_yticklabels(tick_labels, fontsize=7)
    
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else f"{title.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_shortest_path_heatmap_multiple(
    graphs_dict: dict[str, nx.Graph | nx.DiGraph],
    *,
    node_order: list | None = None,
    num_endpoints: int | None = None,
    num_ticks: int = 42,
    x_label: str = "Target Node",
    y_label: str = "Source Node",
    save_dir: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    dpi: int = 300,
):
    """
    Generate shortest path length heatmaps for multiple graphs.
    
    Args:
        graphs_dict: Dictionary mapping labels to NetworkX graphs.
        node_order: Optional list specifying the order of nodes (shared across all graphs).
        num_endpoints: If set, only include nodes 0 to num_endpoints-1 (GPU/endpoint nodes).
        num_ticks: Number of ticks to show on x and y axes (default: 42).
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        save_dir: Directory to save plots. If None, shows interactively.
        figsize: Figure size as (width, height).
        dpi: Resolution for saved figures.
    """
    if not graphs_dict:
        print("No graphs provided.")
        return
    
    for label, G in graphs_dict.items():
        title = f"{label} – Shortest Path Heatmap"
        filename = f"{label.replace(' ', '_')}_heatmap.png" if save_dir else None
        
        plot_shortest_path_heatmap(
            G,
            title=title,
            node_order=node_order,
            num_endpoints=num_endpoints,
            num_ticks=num_ticks,
            x_label=x_label,
            y_label=y_label,
            save_dir=save_dir,
            filename=filename,
            figsize=figsize,
            dpi=dpi,
        )
        
        if save_dir:
            print(f"✅ Saved heatmap for {label}")