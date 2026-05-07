"""
Helper utilities for graph-overloading experiments:
- load assignment/routing helpers
- delay and edge-load calculations
- plotting and CSV export helpers
"""

import random
from collections import defaultdict, deque
from typing import Dict, Tuple, Iterable, List
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import math
import os
import glob
import re
from pathlib import Path


def _is_gpu(node: object, num_endpoints: int) -> bool:
    return isinstance(node, int) and 0 <= node < num_endpoints


def _shortest_path_no_gpu_transit(
    G: nx.DiGraph,
    s: int,
    t: int,
    *,
    num_endpoints: int,
    weight: str = "weight",
) -> List[int]:
    """
    One shortest path from s to t where GPUs (0..num_endpoints-1) cannot be intermediate.
    Implemented by using a filtered view where GPU nodes have no outgoing edges,
    except the source GPU s.
    """
    def filter_edge(u, v) -> bool:
        # Forbid outgoing edges from any GPU u != s
        if _is_gpu(u, num_endpoints) and u != s:
            return False
        return True

    H = nx.subgraph_view(G, filter_edge=filter_edge)
    return nx.shortest_path(H, source=s, target=t, weight=weight)


def _all_shortest_paths_no_gpu_transit(
    G: nx.DiGraph,
    s: int,
    t: int,
    *,
    num_endpoints: int,
    weight: str = "weight",
) -> Iterable[List[int]]:
    """
    All equal-cost shortest paths from s to t where GPUs cannot be intermediate.
    Uses the same filtered view trick.
    """
    def filter_edge(u, v) -> bool:
        if _is_gpu(u, num_endpoints) and u != s:
            return False
        return True

    H = nx.subgraph_view(G, filter_edge=filter_edge)
    return nx.all_shortest_paths(H, source=s, target=t, weight=weight)


def assign_od_to_edges_shortest(
    G: nx.DiGraph,
    OD_csr,
    *,
    num_endpoints: int,
    weight: str = "weight",
    split_equal_shortest: bool = False,
) -> Dict[Tuple[int, int], float]:
    """
    Route each nonzero OD demand on shortest paths and accumulate per-edge load,
    while forbidding intermediate GPUs (0..num_endpoints-1).

    - GPUs may be sources and targets.
    - But no path may *leave* any GPU other than the source GPU s.
      (So GPUs cannot forward traffic.)

    If split_equal_shortest=False:
        - Uses ONE shortest path per (s,t) (NetworkX tie-break).
    If split_equal_shortest=True:
        - Splits demand evenly across ALL equal-cost shortest paths (ECMP-style).

    Returns: dict[(u,v)] -> load
    """
    edge_load = defaultdict(float)
    n = OD_csr.shape[0]

    # Optional safety: ensure OD matches endpoints domain
    if n > num_endpoints:
        # You might be including switches in OD; that usually isn't desired.
        # We won't error, just note: only 0..num_endpoints-1 are treated as GPUs.
        pass

    for s in range(n):
        row_start = OD_csr.indptr[s]
        row_end = OD_csr.indptr[s + 1]
        if row_start == row_end:
            continue

        js = OD_csr.indices[row_start:row_end]
        ds = OD_csr.data[row_start:row_end]

        for t, demand in zip(js, ds):
            s_int = int(s)
            t_int = int(t)
            demand_f = float(demand)

            if demand_f <= 0 or s_int == t_int:
                continue

            # If either endpoint isn't actually in the graph, skip
            if s_int not in G or t_int not in G:
                continue

            try:
                if not split_equal_shortest:
                    path = _shortest_path_no_gpu_transit(
                        G, s_int, t_int, num_endpoints=num_endpoints, weight=weight
                    )
                    for u, v in zip(path[:-1], path[1:]):
                        edge_load[(u, v)] += demand_f
                else:
                    paths = list(
                        _all_shortest_paths_no_gpu_transit(
                            G, s_int, t_int, num_endpoints=num_endpoints, weight=weight
                        )
                    )
                    k = len(paths)
                    if k == 0:
                        continue
                    share = demand_f / k
                    for path in paths:
                        for u, v in zip(path[:-1], path[1:]):
                            edge_load[(u, v)] += share

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    return dict(edge_load)



def get_edge_load_stats(G) -> dict:
    """
    Get statistics about edge loads in the graph.
    
    Returns dict with: min, max, mean, median, total, num_edges, num_nonzero
    """
    loads = np.array([float(d.get("load", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)
    if loads.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "total": 0, "num_edges": 0, "num_nonzero": 0}
    
    nonzero = loads[loads > 0]
    return {
        "min": float(np.min(loads)),
        "max": float(np.max(loads)),
        "mean": float(np.mean(loads)),
        "median": float(np.median(loads)),
        "total": float(np.sum(loads)),
        "num_edges": int(loads.size),
        "num_nonzero": int(nonzero.size),
        "p99": float(np.percentile(loads, 99)) if loads.size > 0 else 0,
    }


def get_graph_switch_and_link_counts(
    G: nx.Graph | nx.DiGraph,
    *,
    num_endpoints: int | None = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Report how many switches and links a topology graph has.

    Switch count is inferred in this priority:
      1) G.graph["meta"]["counts"]["total_switches"] (if available)
      2) Topology-specific metadata fields (FatTree/HyperX/etc.)
      3) total_nodes - num_endpoints (if num_endpoints is provided)

    For DiGraph, both directed edge count and undirected-link count are returned.
    """
    total_nodes = int(G.number_of_nodes())
    directed_edges = int(G.number_of_edges())

    # Count physical links (collapse u->v and v->u in directed graphs).
    if G.is_directed():
        undirected_links = len({frozenset((u, v)) for u, v in G.edges() if u != v})
    else:
        undirected_links = directed_edges

    meta = G.graph.get("meta", {}) if isinstance(G.graph, dict) else {}
    counts = meta.get("counts", {}) if isinstance(meta, dict) else {}
    switch_count: int | None = None
    rails = spines = superspines = None

    if isinstance(counts, dict) and "total_switches" in counts:
        switch_count = int(counts["total_switches"])
    elif isinstance(counts, dict) and {"rails", "spines", "superspines"}.issubset(counts.keys()):
        rails = int(counts["rails"])
        spines = int(counts["spines"])
        superspines = int(counts["superspines"])
        switch_count = rails + spines + superspines
    elif isinstance(meta, dict) and isinstance(meta.get("params"), dict) and "routers" in meta["params"]:
        switch_count = int(meta["params"]["routers"])
    elif isinstance(meta, dict) and "num_gpus" in meta:
        switch_count = total_nodes - int(meta["num_gpus"])
    elif isinstance(meta, dict) and "num_endpoints" in meta:
        switch_count = total_nodes - int(meta["num_endpoints"])
    elif num_endpoints is not None:
        switch_count = total_nodes - int(num_endpoints)

    if switch_count is None:
        raise ValueError(
            "Could not infer switch count from graph metadata. "
            "Pass num_endpoints=<number_of_gpus> to compute switches as total_nodes - num_endpoints."
        )

    endpoint_count = total_nodes - switch_count
    result = {
        "switches": int(switch_count),
        "endpoints": int(endpoint_count),
        "links": int(undirected_links),
        "directed_edges": int(directed_edges),
        "total_nodes": int(total_nodes),
    }
    if rails is not None:
        result["rails"] = int(rails)
    if spines is not None:
        result["spines"] = int(spines)
    if superspines is not None:
        result["superspines"] = int(superspines)

    if verbose:
        if rails is not None:
            print(
                f"Switches: {result['switches']} "
                f"(rails={result['rails']}, spines={result['spines']}, superspines={result['superspines']}), "
                f"Links: {result['links']} "
                f"(directed edges: {result['directed_edges']}), "
                f"Endpoints: {result['endpoints']}, Total nodes: {result['total_nodes']}"
            )
        else:
            print(
                f"Switches: {result['switches']}, "
                f"Links: {result['links']} "
                f"(directed edges: {result['directed_edges']}), "
                f"Endpoints: {result['endpoints']}, Total nodes: {result['total_nodes']}"
            )

    return result


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

def _single_source_shortest_path_length_no_gpu_transit(
    G: nx.Graph | nx.DiGraph,
    source: int,
    *,
    num_endpoints: int,
) -> dict[int, int]:
    """
    Unweighted shortest-path lengths where GPU/endpoint nodes (0..num_endpoints-1)
    are NOT allowed as transit nodes.

    - You can start at a GPU (source)
    - You can end at a GPU (target)
    - But if you reach some other GPU u != source, you do NOT expand from it.
    """
    is_gpu = lambda x: isinstance(x, int) and 0 <= x < num_endpoints

    dist: dict[int, int] = {source: 0}
    q = deque([source])

    while q:
        u = q.popleft()
        du = dist[u]

        # If u is a GPU and not the source, it cannot forward traffic.
        if is_gpu(u) and u != source:
            continue

        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)

    return dist

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

    plt.figure(figsize=(14, 6))
    plt.plot(loads, y)
    plt.ylabel("Edges with load ≤ x (%)")
    plt.xlabel("Edge load (bytes in matrix window)")
    # title = "Edge Load Comparison over 1024 GPUs"
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
    If save_dir is provided, the CSV stores the exact (load, cdf_percent) points
    used for plotting, including the left anchor segment.

    Args:
        graphs_dict: Dictionary mapping string labels to NetworkX graphs. Each edge must have attribute 'load' (float).
        title: Plot title.
        use_log_x: Whether to use log scale on x-axis.
        include_zeros: Whether to include zero loads in the calculation.
    """
    if not graphs_dict:
        print("No graphs provided.")
        return

    plt.figure(figsize=(14, 6))
    
    # Use matplotlib's default color cycle (same as histogram)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
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
    
    # Keep exact plotted points so saved CSV reproduces the rendered curves.
    plotted_rows = []
    
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
        color = default_colors[idx % len(default_colors)]
        
        # Compute marker indices that are evenly spaced in log-space (for log x-axis)
        num_markers = 30
        if use_log_x and len(plot_x) > num_markers:
            # Filter to positive x values for log spacing
            positive_mask = plot_x > 0
            if np.any(positive_mask):
                log_x = np.log10(np.maximum(plot_x, 1e-12))
                log_min, log_max = log_x[positive_mask].min(), log_x[positive_mask].max()
                if log_max > log_min:
                    # Generate evenly spaced positions in log-space
                    log_targets = np.linspace(log_min, log_max, num_markers)
                    # Find nearest data point index for each target
                    marker_indices = []
                    for target in log_targets:
                        idx_nearest = np.argmin(np.abs(log_x - target))
                        if idx_nearest not in marker_indices:
                            marker_indices.append(idx_nearest)
                    markevery = marker_indices
                else:
                    markevery = max(1, len(plot_x) // num_markers)
            else:
                markevery = max(1, len(plot_x) // num_markers)
        else:
            markevery = max(1, len(plot_x) // num_markers)
        
        plt.plot(plot_x, plot_y, color=color, label=label, linewidth=2, 
                 marker=marker, markersize=5, markevery=markevery)

        for point_idx, (xv, yv) in enumerate(zip(plot_x, plot_y)):
            plotted_rows.append({
                "topology": label,
                "point_idx": int(point_idx),
                "load": float(xv),
                "cdf_percent": float(yv),
                "num_edges": int(loads.size),
                "num_zeros": int(num_zeros),
                "first_positive_load": float(first_positive),
                "global_x_start": float(global_x_start),
            })

    plt.ylabel("Edges with load ≤ x (%)", fontsize=16)
    plt.xlabel("Edge load (bytes in matrix window)", fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=13)

    if use_log_x:
        # log scale helps when loads span orders of magnitude
        plt.xscale("log")

    plt.ylim(0, 105)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else f"{title}.png"
        plt.savefig(os.path.join(save_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close()
        
        # Save exact plotted CDF curve data as CSV (matches what was rendered).
        csv_out_name = out_name.replace(".png", ".csv")
        csv_path = os.path.join(save_dir, csv_out_name)
        pd.DataFrame(plotted_rows).to_csv(csv_path, index=False)
        print(f"📄 Saved CDF data CSV: {csv_path}")
    else:
        plt.show()


def plot_average_cdf_from_csvs(
    csv_dir: str,
    *,
    world_size: str,
    title: str | None = None,
    use_log_x: bool = True,
    save_dir: str | None = None,
    filename: str | None = None,
) -> None:
    """
    Load all CDF CSV files from csv_dir that match the given world_size,
    average the exact plotted curve points per topology, and plot the averaged CDF.
    Preferred input schema is the one saved by plot_edge_load_cdf_multiple:
      [topology, point_idx, load, cdf_percent]
    Legacy sampled schemas are also supported.
    
    Args:
        csv_dir: Path to directory containing cdf_*.csv files.
        world_size: World size to filter by (e.g., "1024"). Only CSVs with
                    "world_size{world_size}" in filename are included.
        title: Plot title. If None, auto-generated.
        use_log_x: Whether to use log scale on x-axis.
        save_dir: Directory to save the plot. If None, uses csv_dir.
        filename: Output filename. If None, auto-generated.
    """
    csv_dir = os.path.expanduser(str(csv_dir))
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"Directory not found: {csv_dir!r}")
    
    # Find all CDF CSVs matching the world size
    world_size_pattern = f"world_size{world_size}"
    pattern = os.path.join(csv_dir, "cdf_*.csv")
    all_files = sorted(glob.glob(pattern))
    
    files = [f for f in all_files if world_size_pattern in os.path.basename(f)]
    
    if not files:
        raise FileNotFoundError(
            f"No cdf_*.csv files found with '{world_size_pattern}' in filename. "
            f"Found {len(all_files)} total CDF CSVs in {csv_dir!r}"
        )
    
    print(f"📂 Found {len(files)} CDF CSV(s) with world_size={world_size}")
    
    # Load all per-workload CSV curves and normalize to a common schema:
    # [topology, point_idx, load, cdf_percent]
    # Then average point-by-point on (topology, point_idx).
    normalized_parts: List[pd.DataFrame] = []

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ⚠️ Skipping {os.path.basename(csv_path)}: {e}")
            continue

        # New exact-curve format (preferred)
        if {"topology", "point_idx", "load", "cdf_percent"}.issubset(df.columns):
            part = df[["topology", "point_idx", "load", "cdf_percent"]].copy()
            part["point_idx"] = part["point_idx"].astype(int)
            part["load"] = part["load"].astype(float)
            part["cdf_percent"] = part["cdf_percent"].astype(float)
            normalized_parts.append(part)
            continue

        # Legacy sampled format
        if {"topology", "load", "cdf_percent"}.issubset(df.columns):
            part = df[["topology", "load", "cdf_percent"]].copy()
            part["load"] = part["load"].astype(float)
            part["cdf_percent"] = part["cdf_percent"].astype(float)
            part = part.sort_values(["topology", "cdf_percent", "load"]).reset_index(drop=True)
            part["point_idx"] = part.groupby("topology").cumcount()
            normalized_parts.append(part[["topology", "point_idx", "load", "cdf_percent"]])
            continue

        if {"topology", "percentile", "load"}.issubset(df.columns):
            part = df[["topology", "percentile", "load"]].copy()
            part["load"] = part["load"].astype(float)
            part["cdf_percent"] = part["percentile"].astype(float)
            part = part.sort_values(["topology", "cdf_percent", "load"]).reset_index(drop=True)
            part["point_idx"] = part.groupby("topology").cumcount()
            normalized_parts.append(part[["topology", "point_idx", "load", "cdf_percent"]])
            continue

        print(f"  ⚠️ Skipping {os.path.basename(csv_path)}: unsupported CSV schema")

    if not normalized_parts:
        raise RuntimeError("No valid data found in any CSV files")

    all_df = pd.concat(normalized_parts, ignore_index=True)

    # Average exact plotted points by topology and point index.
    grouped = (
        all_df.groupby(["topology", "point_idx"], as_index=False)
        .agg(
            avg_load=("load", "mean"),
            avg_cdf_percent=("cdf_percent", "mean"),
            count=("load", "count"),
        )
    )

    # For log-x plotting of zeros, track raw minimum positive load per topology.
    topo_min_positive_raw: Dict[str, float] = {}
    for topo_name, sub in all_df.groupby("topology"):
        positive = sub.loc[sub["load"] > 0, "load"]
        if not positive.empty:
            topo_min_positive_raw[topo_name] = float(positive.min())

    topo_avg_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for topo_name, sub in grouped.groupby("topology"):
        sub = sub.sort_values("point_idx")
        loads_arr = sub["avg_load"].to_numpy(dtype=float)
        cdf_arr = sub["avg_cdf_percent"].to_numpy(dtype=float)
        # Enforce monotonic non-decreasing curves:
        # each point is max(current averaged value, previous point).
        loads_arr = np.maximum.accumulate(loads_arr)
        cdf_arr = np.maximum.accumulate(np.clip(cdf_arr, 0.0, 100.0))
        topo_avg_data[topo_name] = (loads_arr, cdf_arr)
    
    # Generate title
    if title is None:
        title = f"Average Edge-load CDF ({world_size} GPUs, {len(files)} workloads)"
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '<', '>', '8', 'P', 'X']
    
    # Calculate global x-axis range with padding (for plotted values)
    all_loads = []
    for topo_name, (loads, _) in topo_avg_data.items():
        if use_log_x:
            topo_min = topo_min_positive_raw.get(topo_name, None)
            if topo_min is not None and topo_min > 0:
                loads_for_plot = np.where(loads > 0, loads, topo_min / 2.0)
            else:
                loads_for_plot = np.where(loads > 0, loads, 1.0)
        else:
            loads_for_plot = loads
        positive_loads = loads_for_plot[loads_for_plot > 0]
        if len(positive_loads) > 0:
            all_loads.extend(positive_loads)
    
    if len(all_loads) > 0:
        x_min = min(all_loads)
        x_max = max(all_loads)
        if use_log_x:
            # For log scale, extend by factors
            x_min = x_min / 2  # Extend left
            x_max = x_max * 2   # Extend right
        else:
            # For linear scale, extend by percentage
            x_range = x_max - x_min
            x_min = max(0, x_min - x_range * 0.1)  # Extend left by 10%
            x_max = x_max + x_range * 0.1          # Extend right by 10%
    else:
        x_min = 0.0
        x_max = 1.0
    
    topo_order = ["Fat Tree", "HyperX", "Dragonfly+"]
    ordered_topologies = [t for t in topo_order if t in topo_avg_data]
    ordered_topologies.extend(sorted([t for t in topo_avg_data if t not in topo_order]))
    
    for idx, topo_name in enumerate(ordered_topologies):
        loads, pcts = topo_avg_data[topo_name]
        marker = markers[idx % len(markers)]
        color = default_colors[idx % len(default_colors)]
        if use_log_x:
            topo_min = topo_min_positive_raw.get(topo_name, None)
            if topo_min is not None and topo_min > 0:
                plot_loads = np.where(loads > 0, loads, topo_min / 2.0)
            else:
                plot_loads = np.where(loads > 0, loads, 1.0)
        else:
            plot_loads = loads

        # Spread markers visually evenly across x-axis (especially on log scale).
        num_markers = 30
        if len(plot_loads) > num_markers:
            if use_log_x:
                positive_mask = plot_loads > 0
                if np.any(positive_mask):
                    log_x = np.log10(np.maximum(plot_loads, 1e-12))
                    log_min = log_x[positive_mask].min()
                    log_max = log_x[positive_mask].max()
                    if log_max > log_min:
                        log_targets = np.linspace(log_min, log_max, num_markers)
                        marker_indices = []
                        for target in log_targets:
                            idx_nearest = int(np.argmin(np.abs(log_x - target)))
                            if idx_nearest not in marker_indices:
                                marker_indices.append(idx_nearest)
                        markevery = marker_indices
                    else:
                        markevery = max(1, len(plot_loads) // num_markers)
                else:
                    markevery = max(1, len(plot_loads) // num_markers)
            else:
                markevery = max(1, len(plot_loads) // num_markers)
        else:
            markevery = 1

        plt.plot(
            plot_loads,
            pcts,
            color=color,
            label=topo_name,
            linewidth=2,
            marker=marker,
            markersize=6,
            markevery=markevery,
        )
    
    plt.ylabel("CDF (%)", fontsize=18)
    plt.xlabel("Edge load (bytes in matrix window)", fontsize=18)
    plt.title(title, fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if use_log_x:
        plt.xscale("log")
        plt.xlim(x_min, x_max)
    else:
        plt.xlim(x_min, x_max)
    
    # Y-axis from 0 to 100 (with small padding for visual clarity)
    plt.ylim(-2, 102)
    
    # Determine save location
    if save_dir is None:
        save_dir = csv_dir
    
    if filename is None:
        filename = f"avg_cdf_{world_size}gpus.png"
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Saved average CDF plot: {out_path}")
    
    # Also save the averaged data as CSV (aligned with plotted curve points)
    csv_out_path = out_path.replace(".png", ".csv")
    csv_rows = []
    for topo_name in ordered_topologies:
        loads, pcts = topo_avg_data[topo_name]
        sub = grouped[grouped["topology"] == topo_name].set_index("point_idx")
        for point_idx, (load, pct) in enumerate(zip(loads, pcts)):
            cnt = int(sub.loc[point_idx, "count"]) if point_idx in sub.index else 0
            csv_rows.append({
                "topology": topo_name,
                "point_idx": int(point_idx),
                "cdf_percent": float(pct),
                "avg_load": float(load),
                "count": cnt,
            })
    
    pd.DataFrame(csv_rows).to_csv(csv_out_path, index=False)
    print(f"📄 Saved average CDF data CSV: {csv_out_path}")

    # Also save "best topology over time" analysis on a common load axis.
    # "Time" is represented by load (x-axis): topology with higher CDF% is better.
    analysis_topos = [t for t in ["Fat Tree", "HyperX", "Dragonfly+"] if t in topo_avg_data]
    if len(analysis_topos) >= 2:
        interp_inputs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        grid_values: List[float] = []

        for topo_name in analysis_topos:
            loads, pcts = topo_avg_data[topo_name]
            # Keep finite values only
            valid = np.isfinite(loads) & np.isfinite(pcts)
            loads = loads[valid]
            pcts = np.clip(pcts[valid], 0.0, 100.0)
            if loads.size == 0:
                continue

            # Ensure monotonicity and collapse duplicate x values.
            loads = np.maximum.accumulate(loads)
            pcts = np.maximum.accumulate(pcts)
            uniq_loads, inv = np.unique(loads, return_inverse=True)
            uniq_pcts = np.zeros_like(uniq_loads, dtype=float)
            for i in range(len(uniq_loads)):
                uniq_pcts[i] = float(np.max(pcts[inv == i]))

            interp_inputs[topo_name] = (uniq_loads, uniq_pcts)
            grid_values.extend(uniq_loads.tolist())

        if grid_values:
            common_load_grid = np.array(sorted(set(grid_values)), dtype=float)
            eps = 1e-9
            reached_100: set[str] = set()
            analysis_rows: List[Dict[str, object]] = []

            for load_x in common_load_grid:
                pct_at_load: Dict[str, float] = {}
                for topo_name in analysis_topos:
                    if topo_name not in interp_inputs:
                        continue
                    x_arr, y_arr = interp_inputs[topo_name]
                    if x_arr.size == 1:
                        pct_val = float(y_arr[0])
                    else:
                        pct_val = float(np.interp(load_x, x_arr, y_arr, left=y_arr[0], right=y_arr[-1]))
                    pct_at_load[topo_name] = float(np.clip(pct_val, 0.0, 100.0))

                if not pct_at_load:
                    continue

                vals = list(pct_at_load.values())
                at_100_now = [t for t, v in pct_at_load.items() if v >= 100.0 - eps]

                if all(abs(v) <= eps for v in vals):
                    # Rule 1: all are zero -> write all.
                    winners = list(analysis_topos)
                    rule = "all_zero_write_all"
                elif at_100_now:
                    # Rules 3-4: once someone reaches 100, keep first-to-reach set,
                    # adding any additional topologies when they also reach 100.
                    reached_100.update(at_100_now)
                    winners = [t for t in analysis_topos if t in reached_100]
                    rule = "reached_100_first_to_last"
                else:
                    # Rule 2: while nobody reached 100, choose highest percentage.
                    best_val = max(vals)
                    winners = [t for t, v in pct_at_load.items() if abs(v - best_val) <= eps]
                    rule = "highest_below_100"

                row: Dict[str, object] = {
                    "load": float(load_x),
                    "best_topology": ";".join(winners),
                    "rule_applied": rule,
                }
                for topo_name in analysis_topos:
                    row[f"{topo_name}_cdf_percent"] = float(pct_at_load.get(topo_name, np.nan))
                analysis_rows.append(row)

            if analysis_rows:
                analysis_out_path = out_path.replace(".png", "_best_topology_over_time.csv")
                pd.DataFrame(analysis_rows).to_csv(analysis_out_path, index=False)
                print(f"📄 Saved best-topology-over-time CSV: {analysis_out_path}")

                # Also summarize domination share on a linear load scale.
                # Weight each stage by its linear interval [load_i, load_{i+1}) and split ties equally.
                loads_seq = np.array([float(r["load"]) for r in analysis_rows], dtype=float)
                n_rows = int(loads_seq.size)
                if n_rows == 1:
                    # Single stage: assign full unit weight.
                    stage_weights = np.array([1.0], dtype=float)
                else:
                    # Assign each row i the span to the next row.
                    # Last row has no interval after it, so its weight is 0.
                    stage_weights = np.zeros(n_rows, dtype=float)
                    diffs = np.diff(loads_seq)
                    stage_weights[:-1] = np.maximum(0.0, diffs)
                    if float(stage_weights.sum()) <= 0.0:
                        stage_weights = np.ones(n_rows, dtype=float)

                weighted_share = {t: 0.0 for t in analysis_topos}
                inclusive_share = {t: 0.0 for t in analysis_topos}
                total_weight = float(stage_weights.sum())

                for i, row in enumerate(analysis_rows):
                    winners = [w for w in str(row["best_topology"]).split(";") if w]
                    if not winners:
                        continue
                    w = float(stage_weights[i])
                    split = w / float(len(winners))
                    for winner in winners:
                        if winner in weighted_share:
                            weighted_share[winner] += split
                            inclusive_share[winner] += w

                dominance_rows: List[Dict[str, object]] = []
                for topo_name in analysis_topos:
                    w_raw = float(weighted_share[topo_name])
                    i_raw = float(inclusive_share[topo_name])
                    dominance_rows.append(
                        {
                            "topology": topo_name,
                            "dominance_percent_linear_scale": (
                                100.0 * w_raw / total_weight if total_weight > 0 else 0.0
                            ),
                            "dominance_percent_linear_scale_inclusive_ties": (
                                100.0 * i_raw / total_weight if total_weight > 0 else 0.0
                            ),
                            "dominance_weight_raw": w_raw,
                            "inclusive_weight_raw": i_raw,
                            "total_weight_raw": total_weight,
                            "num_stages": n_rows,
                            "weighting_method": "linear_interval_right_open",
                        }
                    )

                dominance_out_path = out_path.replace(".png", "_best_topology_dominance_linear.csv")
                pd.DataFrame(dominance_rows).to_csv(dominance_out_path, index=False)
                print(f"📄 Saved best-topology dominance CSV: {dominance_out_path}")

                # Also summarize domination share on experiment-linear scale
                # (each stage interval contributes equally, independent of load span).
                exp_weighted_share = {t: 0.0 for t in analysis_topos}
                exp_inclusive_share = {t: 0.0 for t in analysis_topos}
                # Right-open intervals [i, i+1): last row has no interval.
                n_intervals = max(1, n_rows - 1)
                exp_stage_weights = np.zeros(n_rows, dtype=float)
                if n_rows == 1:
                    exp_stage_weights[0] = 1.0
                else:
                    exp_stage_weights[:-1] = 1.0

                exp_total_weight = float(exp_stage_weights.sum())
                for i, row in enumerate(analysis_rows):
                    winners = [w for w in str(row["best_topology"]).split(";") if w]
                    if not winners:
                        continue
                    w = float(exp_stage_weights[i])
                    if w <= 0:
                        continue
                    split = w / float(len(winners))
                    for winner in winners:
                        if winner in exp_weighted_share:
                            exp_weighted_share[winner] += split
                            exp_inclusive_share[winner] += w

                exp_rows: List[Dict[str, object]] = []
                for topo_name in analysis_topos:
                    w_raw = float(exp_weighted_share[topo_name])
                    i_raw = float(exp_inclusive_share[topo_name])
                    exp_rows.append(
                        {
                            "topology": topo_name,
                            "dominance_percent_experiment_linear": (
                                100.0 * w_raw / exp_total_weight if exp_total_weight > 0 else 0.0
                            ),
                            "dominance_percent_experiment_linear_inclusive_ties": (
                                100.0 * i_raw / exp_total_weight if exp_total_weight > 0 else 0.0
                            ),
                            "dominance_weight_raw": w_raw,
                            "inclusive_weight_raw": i_raw,
                            "total_weight_raw": exp_total_weight,
                            "num_stages": n_rows,
                            "num_intervals": n_intervals,
                            "weighting_method": "experiment_stage_linear_right_open",
                        }
                    )

                exp_out_path = out_path.replace(".png", "_best_topology_dominance_experiment_linear.csv")
                pd.DataFrame(exp_rows).to_csv(exp_out_path, index=False)
                print(f"📄 Saved experiment-linear dominance CSV: {exp_out_path}")


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
        percentiles = [10, 25, 50, 60, 70, 80, 90, 95, 99]
    
    percentiles = sorted(percentiles)
    
    plt.figure(figsize=(12, 7))
    
    # Use matplotlib's default color cycle (same as histogram)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
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
        color = default_colors[idx % len(default_colors)]
        
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
            if num_endpoints is not None and isinstance(source, int) and 0 <= source < num_endpoints:
                lengths = _single_source_shortest_path_length_no_gpu_transit(
                    G, source, num_endpoints=num_endpoints
                )
            else:
                # If you're not filtering to endpoints, fall back to vanilla behavior
                lengths = nx.single_source_shortest_path_length(G, source)

            src_idx = node_to_idx[source]
            for target, length in lengths.items():
                if target in node_to_idx:
                    tgt_idx = node_to_idx[target]
                    dist_matrix[src_idx, tgt_idx] = length
        except nx.NetworkXError:
            continue
    
    # Determine discrete values present in the matrix
    valid_values = dist_matrix[~np.isnan(dist_matrix)]
    if valid_values.size == 0:
        print("No reachable paths in the graph.")
        return
    
    min_dist = int(np.min(valid_values))
    max_dist = int(np.max(valid_values))
    # Ensure at least 5 discrete color values
    min_colors = 7
    if max_dist - min_dist + 1 < min_colors:
        max_dist = min_dist + min_colors - 1
    discrete_values = list(range(min_dist, max_dist + 1))
    
    # Create discrete colormap: cold (low) to warm (high)
    # 7-color scale from deep blue to dark red
    cold_to_warm = [
        '#313695',  # deep navy (coldest / strongest negative)
        '#74add1',  # distinct mid-blue
        '#e0f3f8',  # pale icy blue (near zero)
        '#fee090',  # pale warm yellow (near zero)
        '#f46d43',  # vibrant orange-red
        '#a50026',  # deep crimson (warmest / strongest positive)
    ]
    # Select colors based on number of discrete values needed
    num_colors = len(discrete_values)
    if num_colors <= len(cold_to_warm):
        # Pick evenly spaced colors from our palette
        indices = np.linspace(0, len(cold_to_warm) - 1, num_colors).astype(int)
        colors = [cold_to_warm[i] for i in indices]
    else:
        # Fall back to coolwarm colormap for more colors
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_colors))
    cmap = ListedColormap(colors)
    
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
        
        # Save as CSV with GPU/node indices
        csv_name = out_name.replace(".png", ".csv")
        gpu_labels = [f"GPU{node}" for node in node_order]
        df_heatmap = pd.DataFrame(dist_matrix, index=gpu_labels, columns=gpu_labels)
        df_heatmap.to_csv(os.path.join(save_dir, csv_name))
        print(f"📄 Saved CSV: {os.path.join(save_dir, csv_name)}")
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


def save_effective_heatmap_csv(
    heatmaps_dir: str,
    transport_dir: str,
    *,
    save_dir: str | None = None,
) -> List[str]:
    """
    Batch-create "effective heatmap" CSVs by masking each heatmap with its matching
    transport matrix.

    For each heatmap CSV in `heatmaps_dir`:
      - infer workload name from heatmap filename (Topology_workload.csv)
      - find matching transport CSV in `transport_dir` by workload stem
      - keep heatmap[i, j] if transport[i, j] != 0, else set to 0
      - save as: effective_heatmap_<original_heatmap_filename>.csv

    Supports both labeled square CSVs (row/col labels) and plain numeric square CSVs.

    Args:
        heatmaps_dir: Directory containing heatmap CSV files.
        transport_dir: Directory containing transport-matrix CSV files (workload-named).
        save_dir: Output directory (defaults to heatmaps_dir).

    Returns:
        List of output CSV paths that were saved.
    """
    def _load_square_csv(path: str) -> pd.DataFrame:
        # Try labeled CSV first (index column + header row)
        try:
            df = pd.read_csv(path, index_col=0)
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if df.shape[0] == df.shape[1] and df.shape[0] > 0:
                return df
        except Exception:
            pass

        # Fallback: plain numeric CSV with no labels
        arr = np.loadtxt(path, delimiter=",", dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"CSV must be square matrix: {path!r}, got shape={arr.shape}")
        labels = [f"GPU{i}" for i in range(arr.shape[0])]
        return pd.DataFrame(arr, index=labels, columns=labels)

    heatmaps_dir = os.path.expanduser(str(heatmaps_dir))
    transport_dir = os.path.expanduser(str(transport_dir))
    if not os.path.isdir(heatmaps_dir):
        raise FileNotFoundError(f"heatmaps_dir not found: {heatmaps_dir!r}")
    if not os.path.isdir(transport_dir):
        raise FileNotFoundError(f"transport_dir not found: {transport_dir!r}")

    heatmap_files = sorted(glob.glob(os.path.join(heatmaps_dir, "*.csv")))
    if not heatmap_files:
        raise FileNotFoundError(f"No heatmap CSV files found in {heatmaps_dir!r}")

    transport_files = sorted(glob.glob(os.path.join(transport_dir, "*.csv")))
    transport_map = {Path(p).stem: p for p in transport_files}
    if not transport_map:
        raise FileNotFoundError(f"No transport CSV files found in {transport_dir!r}")

    if save_dir is None:
        save_dir = heatmaps_dir
    os.makedirs(save_dir, exist_ok=True)

    topo_prefixes = [
        "Fat_Tree_",
        "Fat Tree_",
        "Dragonfly+_",
        "HyperX_",
        "Rail_Only_",
        "Rail Only_",
    ]

    def _extract_workload_stem(heatmap_stem: str) -> str:
        for pref in topo_prefixes:
            if heatmap_stem.startswith(pref):
                return heatmap_stem[len(pref):]
        # Fallback: split once at the first underscore.
        parts = heatmap_stem.split("_", 1)
        return parts[1] if len(parts) == 2 else heatmap_stem

    out_paths: List[str] = []
    for heatmap_path in heatmap_files:
        heatmap_stem = Path(heatmap_path).stem
        workload_stem = _extract_workload_stem(heatmap_stem)
        transport_path = transport_map.get(workload_stem)
        if transport_path is None:
            print(
                f"  ⚠️ Skipping {os.path.basename(heatmap_path)}: "
                f"no matching transport CSV for workload stem {workload_stem!r}"
            )
            continue

        out_name = f"effective_heatmap_{os.path.basename(heatmap_path)}"
        out_path = os.path.join(save_dir, out_name)
        if os.path.isfile(out_path):
            out_paths.append(out_path)
            print(f"⏭️ Skipping (already exists): {out_path}")
            continue

        heatmap_df = _load_square_csv(heatmap_path)
        transport_df = _load_square_csv(transport_path)

        # Prefer label-based alignment when possible.
        if (
            set(heatmap_df.index) == set(transport_df.index)
            and set(heatmap_df.columns) == set(transport_df.columns)
        ):
            transport_aligned = transport_df.loc[heatmap_df.index, heatmap_df.columns]
        elif heatmap_df.shape == transport_df.shape:
            # Fallback: positional alignment if dimensions match but labels don't.
            transport_aligned = pd.DataFrame(
                transport_df.to_numpy(dtype=float, copy=False),
                index=heatmap_df.index,
                columns=heatmap_df.columns,
            )
        else:
            print(
                f"  ⚠️ Skipping {os.path.basename(heatmap_path)}: "
                f"shape mismatch heatmap={heatmap_df.shape} transport={transport_df.shape}"
            )
            continue

        heatmap_vals = heatmap_df.to_numpy(dtype=float, copy=True)
        transport_vals = transport_aligned.to_numpy(dtype=float, copy=False)
        heatmap_vals[transport_vals == 0] = 0.0
        effective_df = pd.DataFrame(heatmap_vals, index=heatmap_df.index, columns=heatmap_df.columns)

        effective_df.to_csv(out_path)
        out_paths.append(out_path)
        print(f"📄 Saved effective heatmap CSV: {out_path}")

    if not out_paths:
        raise RuntimeError(
            "No effective heatmaps were generated (no matches or all pairs failed alignment)."
        )

    return out_paths


def save_effective_heatmap_nonzero_distribution_csv(
    effective_heatmaps_dir: str,
    *,
    save_dir: str | None = None,
    filename: str = "effective_heatmap_nonzero_distribution.csv",
) -> str:
    """
    Aggregate non-zero effective-heatmap values across all CSVs in a directory and
    save a distribution CSV for Fat Tree, HyperX, and Dragonfly+.

    Output columns:
      - topology
      - value
      - count
      - pct_of_topology_nonzero
      - total_nonzero_topology
      - num_files_topology
    """
    effective_heatmaps_dir = os.path.expanduser(str(effective_heatmaps_dir))
    if not os.path.isdir(effective_heatmaps_dir):
        raise FileNotFoundError(f"effective_heatmaps_dir not found: {effective_heatmaps_dir!r}")

    csv_files = sorted(glob.glob(os.path.join(effective_heatmaps_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {effective_heatmaps_dir!r}")

    target_topos = ("Fat Tree", "HyperX", "Dragonfly+")
    topo_prefixes = {
        "Fat Tree": ("effective_heatmap_Fat_Tree_", "effective_heatmap_Fat Tree_"),
        "HyperX": ("effective_heatmap_HyperX_",),
        "Dragonfly+": ("effective_heatmap_Dragonfly+_",),
    }

    topo_values: Dict[str, List[np.ndarray]] = {t: [] for t in target_topos}
    topo_file_counts: Dict[str, int] = {t: 0 for t in target_topos}
    # Per-world-size aggregations: world_size -> topology -> list of non-zero arrays
    topo_values_by_ws: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
        lambda: {t: [] for t in target_topos}
    )
    topo_file_counts_by_ws: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {t: 0 for t in target_topos}
    )

    for fp in csv_files:
        stem = Path(fp).stem
        topo_name = None
        for topo, prefixes in topo_prefixes.items():
            if any(stem.startswith(pref) for pref in prefixes):
                topo_name = topo
                break
        if topo_name is None:
            continue

        # Load matrix (labeled or numeric)
        try:
            df = pd.read_csv(fp, index_col=0)
            arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError("not square")
        except Exception:
            arr = np.loadtxt(fp, delimiter=",", dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                print(f"  ⚠️ Skipping {os.path.basename(fp)}: could not parse square matrix")
                continue

        flat = arr.reshape(-1)
        flat = flat[~np.isnan(flat)]
        nonzero = flat[flat != 0.0]
        if nonzero.size == 0:
            continue

        topo_values[topo_name].append(nonzero)
        topo_file_counts[topo_name] += 1
        ws_match = re.search(r"world_size(\d+)", stem)
        if ws_match is not None:
            world_size = ws_match.group(1)
            topo_values_by_ws[world_size][topo_name].append(nonzero)
            topo_file_counts_by_ws[world_size][topo_name] += 1

    rows = []
    for topo in target_topos:
        if not topo_values[topo]:
            continue
        vals = np.concatenate(topo_values[topo])
        unique_vals, counts = np.unique(vals, return_counts=True)
        total_nonzero = int(np.sum(counts))
        for v, c in zip(unique_vals, counts):
            rows.append(
                {
                    "topology": topo,
                    "value": float(v),
                    "count": int(c),
                    "pct_of_topology_nonzero": (float(c) / total_nonzero) * 100.0,
                    "total_nonzero_topology": total_nonzero,
                    "num_files_topology": topo_file_counts[topo],
                }
            )

    if not rows:
        raise RuntimeError(
            "No non-zero values found for Fat Tree / HyperX / Dragonfly+ in effective heatmap CSVs."
        )

    out_df = pd.DataFrame(rows).sort_values(["topology", "value"]).reset_index(drop=True)

    if save_dir is None:
        save_dir = effective_heatmaps_dir
    os.makedirs(save_dir, exist_ok=True)
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"
    out_path = os.path.join(save_dir, filename)
    out_df.to_csv(out_path, index=False)
    print(f"📄 Saved effective heatmap non-zero distribution CSV: {out_path}")

    # Save per-world-size distributions as separate CSV files.
    base_name = filename[:-4] if filename.lower().endswith(".csv") else filename
    for world_size in sorted(topo_values_by_ws.keys(), key=lambda x: int(x)):
        ws_rows = []
        ws_vals = topo_values_by_ws[world_size]
        ws_counts = topo_file_counts_by_ws[world_size]

        for topo in target_topos:
            if not ws_vals[topo]:
                continue
            vals = np.concatenate(ws_vals[topo])
            unique_vals, counts = np.unique(vals, return_counts=True)
            total_nonzero = int(np.sum(counts))
            for v, c in zip(unique_vals, counts):
                ws_rows.append(
                    {
                        "world_size": int(world_size),
                        "topology": topo,
                        "value": float(v),
                        "count": int(c),
                        "pct_of_topology_nonzero": (float(c) / total_nonzero) * 100.0,
                        "total_nonzero_topology": total_nonzero,
                        "num_files_topology": ws_counts[topo],
                    }
                )

        if not ws_rows:
            continue
        ws_df = pd.DataFrame(ws_rows).sort_values(["topology", "value"]).reset_index(drop=True)
        ws_filename = f"{base_name}_world_size{world_size}.csv"
        ws_out_path = os.path.join(save_dir, ws_filename)
        ws_df.to_csv(ws_out_path, index=False)
        print(f"📄 Saved effective heatmap non-zero distribution CSV (world_size={world_size}): {ws_out_path}")

    return out_path


def compute_gpu_to_gpu_delay_df(
    G: nx.DiGraph,
    OD_csr,
    *,
    num_endpoints: int,
    weight: str = "weight",
    split_equal_shortest: bool = False,
    # Delay model parameters
    bandwidth_bytes_per_sec: float = 4e11,
    alpha_per_hop_sec: float = 0.0,
    # Edge attribute names
    load_attr: str = "load",
    capacity_attr: str = "capacity",
    # Behavior
    include_base_latency_when_zero: bool = False,
    save_dir: str | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Compute an estimated per-(GPU_i -> GPU_j) delay matrix as a DataFrame.

    Purpose
    -------
    Demonstrate contention-driven delay using *routed per-edge loads*.

    Model (per OD entry i->j with bytes S_ij)
    -----------------------------------------
    1) Choose shortest path(s) under the "no GPU transit" rule
       (GPUs 0..num_endpoints-1 cannot be intermediate forwarding nodes).
    2) For each edge e on the chosen path, compute an effective per-byte time:

           t_e_per_byte = (1 / bw_e) * congestion_factor_e

       where bw_e is link bandwidth (bytes/sec). We approximate congestion as:

           congestion_factor_e = 1 / (1 - rho_e)

       using, in priority order:
         - edge attribute 'util' if present (annotate_graph_with_loads sets it)
         - else, load/capacity if capacity is finite
         - else, rho_e = 0

    3) Delay is:

           D_ij = alpha_per_hop_sec * hops_ij + S_ij * sum_{e in path} t_e_per_byte

       If split_equal_shortest=True, we average D_ij uniformly across all equal-cost
       shortest paths (ECMP-style).

    Parameters
    ----------
    bandwidth_bytes_per_sec:
        If provided, used as the bandwidth for ALL edges.
        Otherwise we try to use edge[capacity_attr] if it is finite and >0.
        If neither is available, fall back to 1.0 (normalized units).
    alpha_per_hop_sec:
        Optional per-hop fixed latency term.
    include_base_latency_when_zero:
        If True, pairs with S_ij==0 get alpha*hops (otherwise 0).
    save_dir/filename:
        If save_dir is given, we save the resulting delay DataFrame to CSV.
        We also save a second transport-aligned delay matrix CSV, where entry (i, j)
        is 0 when OD(i, j) == 0, and the computed delay otherwise.
    """

    if num_endpoints <= 0:
        raise ValueError("num_endpoints must be a positive int")

    n = int(num_endpoints)
    delay = np.zeros((n, n), dtype=np.float64)

    def _edge_bw(u: int, v: int) -> float:
        """Return bandwidth (bytes/sec) for edge (u,v)."""
        if bandwidth_bytes_per_sec is not None:
            return float(bandwidth_bytes_per_sec)
        data = G.get_edge_data(u, v, default={})
        cap = data.get(capacity_attr, None)
        try:
            cap_f = float(cap)
        except (TypeError, ValueError):
            cap_f = float("nan")
        if np.isfinite(cap_f) and cap_f > 0:
            return cap_f
        return 1.0

    def _congestion_factor(u: int, v: int) -> float:
        """Return a multiplicative congestion penalty >=1 for edge (u,v)."""
        data = G.get_edge_data(u, v, default={})

        util = data.get("util", None)
        if util is not None:
            try:
                rho = float(util)
            except (TypeError, ValueError):
                rho = 0.0
        else:
            load = float(data.get(load_attr, 0.0) or 0.0)
            cap = data.get(capacity_attr, None)
            try:
                cap_f = float(cap)
            except (TypeError, ValueError):
                cap_f = float("nan")
            if np.isfinite(cap_f) and cap_f > 0:
                rho = load / cap_f
            else:
                rho = 0.0

        # clamp
        if rho < 0:
            rho = 0.0
        if rho >= 1.0:
            rho = 0.999999
        return 1.0 / (1.0 - rho)

    # Iterate non-zeros in OD (CSR assumed)
    for s in range(n):
        row_start = OD_csr.indptr[s]
        row_end = OD_csr.indptr[s + 1]

        # If no traffic, optionally fill base-hop latency
        if row_start == row_end:
            if include_base_latency_when_zero:
                for t in range(n):
                    if s == t:
                        continue
                    try:
                        path = _shortest_path_no_gpu_transit(G, s, t, num_endpoints=n, weight=weight)
                        hops = max(0, len(path) - 1)
                        delay[s, t] = alpha_per_hop_sec * hops
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        delay[s, t] = np.nan
            continue

        js = OD_csr.indices[row_start:row_end]
        ds = OD_csr.data[row_start:row_end]

        for t_raw, bytes_raw in zip(js, ds):
            t = int(t_raw)
            if t < 0 or t >= n or s == t:
                continue

            S = float(bytes_raw)
            if S <= 0:
                if include_base_latency_when_zero:
                    try:
                        path = _shortest_path_no_gpu_transit(G, s, t, num_endpoints=n, weight=weight)
                        hops = max(0, len(path) - 1)
                        delay[s, t] = alpha_per_hop_sec * hops
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        delay[s, t] = np.nan
                else:
                    delay[s, t] = 0.0
                continue

            try:
                if not split_equal_shortest:
                    path = _shortest_path_no_gpu_transit(G, s, t, num_endpoints=n, weight=weight)
                    hops = max(0, len(path) - 1)

                    per_byte = 0.0
                    for u, v in zip(path[:-1], path[1:]):
                        bw = _edge_bw(int(u), int(v))
                        per_byte += (1.0 / bw) * _congestion_factor(int(u), int(v))

                    delay[s, t] = alpha_per_hop_sec * hops + S * per_byte

                else:
                    paths = list(_all_shortest_paths_no_gpu_transit(G, s, t, num_endpoints=n, weight=weight))
                    if not paths:
                        delay[s, t] = np.nan
                        continue

                    acc = 0.0
                    for path in paths:
                        hops = max(0, len(path) - 1)
                        per_byte = 0.0
                        for u, v in zip(path[:-1], path[1:]):
                            bw = _edge_bw(int(u), int(v))
                            per_byte += (1.0 / bw) * _congestion_factor(int(u), int(v))
                        acc += alpha_per_hop_sec * hops + S * per_byte

                    delay[s, t] = acc / float(len(paths))

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                delay[s, t] = np.nan

    labels = [f"GPU{i}" for i in range(n)]
    df = pd.DataFrame(delay, index=labels, columns=labels)

    # Build a transport-aligned delay matrix:
    # - 0 when OD(i,j) == 0
    # - computed delay when OD(i,j) > 0
    # This remains strictly aligned to the original transport matrix sparsity.
    delay_transport_aligned = np.zeros((n, n), dtype=np.float64)
    for s in range(n):
        row_start = OD_csr.indptr[s]
        row_end = OD_csr.indptr[s + 1]
        js = OD_csr.indices[row_start:row_end]
        ds = OD_csr.data[row_start:row_end]
        for t_raw, bytes_raw in zip(js, ds):
            t = int(t_raw)
            if t < 0 or t >= n:
                continue
            if float(bytes_raw) > 0.0:
                delay_transport_aligned[s, t] = delay[s, t]

    df_transport_aligned = pd.DataFrame(
        delay_transport_aligned,
        index=labels,
        columns=labels,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_name = filename if filename is not None else "gpu_to_gpu_delay.csv"
        if not out_name.lower().endswith(".csv"):
            out_name = f"{out_name}.csv"
        out_path = os.path.join(save_dir, out_name)
        df.to_csv(out_path)
        print(f"📄 Saved delay DataFrame CSV: {out_path}")

        base_name = out_name[:-4] if out_name.lower().endswith(".csv") else out_name
        out_path_transport = os.path.join(save_dir, f"{base_name}_transport_aligned.csv")
        df_transport_aligned.to_csv(out_path_transport)
        print(f"📄 Saved transport-aligned delay CSV: {out_path_transport}")

    return df

def load_and_compare_delay_percentiles(
    delay_dir: str,
    *,
    workload_name: str,
    percentiles: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100),
) -> pd.DataFrame:
    """
    From delay_dir, load all delay CSVs whose filename contains workload_name.
    For each CSV:
      - load into a DataFrame
      - flatten to a 1D ascending array containing only non-zero values
      - compute min/max/mean/median
      - compute p% thresholds for all percentiles in `percentiles`
    Save the combined summary table to CSV in delay_dir and return it as a DataFrame.
    """
    delay_dir = os.path.expanduser(str(delay_dir))
    if not os.path.isdir(delay_dir):
        raise FileNotFoundError(f"delay_dir does not exist or is not a directory: {delay_dir!r}")

    # Load all delay CSVs that contain workload_name anywhere in the filename
    # Use *_delay.csv pattern to exclude summary/percentiles files
    pattern = os.path.join(delay_dir, f"*{workload_name}*_delay.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No delay CSVs found in {delay_dir!r} matching pattern {pattern!r}")

    rows = []
    qs = np.array(percentiles, dtype=float) / 100.0

    for fp in files:
        base = os.path.basename(fp)

        # Load
        df = pd.read_csv(fp, index_col=0)
        values = df.to_numpy(dtype=float, copy=False)

        # Flatten + keep only non-zero values
        flat = values.reshape(-1)
        flat = flat[~np.isnan(flat)]  # defensive: drop NaNs if they exist
        flat = flat[flat != 0.0]      # keep only non-zero (can change to >0.0 if you guarantee positivity)
        flat.sort()                   # ascending

        if flat.size == 0:
            # No non-zero entries, skip (or record NaNs; skipping is usually cleaner)
            continue

        # Summary stats
        min_v = float(flat[0])
        max_v = float(flat[-1])
        mean_v = float(np.mean(flat))
        median_v = float(np.median(flat))

        # Percentiles: threshold such that >=p% of samples are <= value
        # Use empirical quantile with "higher" (returns an observed value)
        try:
            pct_vals = np.quantile(flat, qs, method="higher")
        except TypeError:
            pct_vals = np.quantile(flat, qs, interpolation="higher")

        row = {
            "file": base,
            "num_nonzero": int(flat.size),
            "min": min_v,
            "mean": mean_v,
            "median": median_v,
            "max": max_v,
        }
        for p, v in zip(percentiles, pct_vals):
            row[f"p{int(p)}"] = float(v)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"Found {len(files)} CSVs containing {workload_name!r}, but none had any non-zero values."
        )

    out_df = pd.DataFrame(rows)

    # Sort for easier comparison (by median, then p95, then max if present)
    sort_cols = [c for c in ("median", "p95", "max") if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    # Save summary CSV
    safe_workload = workload_name.replace(os.sep, "_")
    out_path = os.path.join(delay_dir, f"delay_summary_{safe_workload}.csv")
    out_df.to_csv(out_path, index=False)

    return out_df


def plot_delay_percentiles_from_csv(
    csv_path: str,
    *,
    title: str | None = None,
    use_log_y: bool = True,
    connect_points: bool = True,
    save_dir: str | None = None,
    filename: str | None = None,
) -> None:
    """
    Generate a percentile plot from a delay summary CSV file.
    
    The CSV is expected to have columns:
      - file: filename (used to extract topology name)
      - p10, p25, p50, p60, p70, p80, p90, p95, p99: percentile values
    
    Args:
        csv_path: Path to the percentile summary CSV file.
        title: Plot title. If None, derived from CSV filename.
        use_log_y: Whether to use log scale on y-axis.
        connect_points: Whether to connect percentile points with lines.
        save_dir: Directory to save the plot. If None, uses same directory as CSV.
        filename: Output filename. If None, derived from CSV filename.
    """
    csv_path = os.path.expanduser(str(csv_path))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path!r}")
    
    df = pd.read_csv(csv_path)
    
    if "file" not in df.columns:
        raise ValueError("CSV must have a 'file' column")
    
    # Find percentile columns (p10, p25, etc.)
    pct_cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    if not pct_cols:
        raise ValueError("No percentile columns (p10, p25, etc.) found in CSV")
    
    # Extract percentile values from column names
    percentiles = sorted([int(c[1:]) for c in pct_cols])
    
    # Extract topology names from filenames
    # Expected format: "TopologyName_workload_delay.csv" -> "TopologyName"
    def extract_topology(filename: str) -> str:
        # Remove _delay.csv suffix if present
        name = filename.replace("_delay.csv", "")
        # Take the part before the first underscore that looks like a workload identifier
        parts = name.split("_")
        # Common topology names
        known_topos = ["Fat_Tree", "Fat Tree", "Dragonfly+", "HyperX", "Rail_Only"]
        for topo in known_topos:
            if name.startswith(topo.replace(" ", "_")):
                return topo.replace("_", " ")
        # Fallback: use first part
        return parts[0].replace("_", " ")
    
    df["topology"] = df["file"].apply(extract_topology)
    
    # Generate title if not provided
    if title is None:
        csv_basename = os.path.basename(csv_path)
        # Extract workload name from "delay_summary_WORKLOAD.csv"
        if csv_basename.startswith("delay_summary_"):
            workload = csv_basename.replace("delay_summary_", "").replace(".csv", "")
            title = f"Delay Percentiles - {workload}"
        else:
            title = f"Delay Percentiles - {csv_basename}"
    
    # Plot setup
    plt.figure(figsize=(12, 7))
    
    # Use matplotlib's default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '<', '>', '8', 'P', 'X']
    
    for idx, (_, row) in enumerate(df.iterrows()):
        topo_name = row["topology"]
        # Convert from seconds to milliseconds
        percentile_values = [row[f"p{p}"] * 1000 for p in percentiles]
        
        marker = markers[idx % len(markers)]
        color = default_colors[idx % len(default_colors)]
        
        if connect_points:
            plt.plot(percentiles, percentile_values, color=color, label=topo_name,
                     linewidth=2, marker=marker, markersize=8)
        else:
            plt.scatter(percentiles, percentile_values, color=color, label=topo_name,
                        marker=marker, s=80)
    
    plt.xlabel("Percentile (%)")
    plt.ylabel("Delay (µs)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper right')  # Move legend to upper right since high values are now on left
    
    # Set x-axis ticks to the percentiles (reversed: highest on left, lowest on right)
    plt.xticks(percentiles)
    plt.xlim(max(percentiles) + 5, min(percentiles) - 5)  # Reversed: high to low
    
    if use_log_y:
        plt.yscale("log")
    
    # Determine save location
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    
    if filename is None:
        csv_basename = os.path.basename(csv_path)
        filename = csv_basename.replace(".csv", ".png").replace("delay_summary_", "delay_percentiles_")
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Saved delay percentiles plot: {out_path}")


def plot_average_delay_percentiles_from_dir(
    csv_dir: str,
    *,
    num_gpus: str | None = None,
    title: str | None = None,
    use_log_y: bool = True,
    connect_points: bool = True,
    save_dir: str | None = None,
    filename: str | None = None,
    normalize: bool = True,
) -> None:
    """
    Generate a percentile plot by averaging delay percentile values across multiple
    delay summary CSV files in a directory.
    
    For each CSV:
      1. Extract topology names from each row
      2. Optionally normalize percentile values by dividing by the max value
         in that CSV (when normalize=True)
      3. Accumulate values per topology
    
    Then compute the average value for each topology at each percentile and plot
    the result.
    
    Args:
        csv_dir: Path to directory containing delay_summary_*.csv files.
        num_gpus: If provided (e.g., "1024"), only include CSVs whose filename
                  contains "world_size{num_gpus}" (e.g., "world_size1024").
                  If None, all delay_summary CSVs are included.
        title: Plot title. If None, auto-generated based on parameters.
        use_log_y: Whether to use log scale on y-axis.
        connect_points: Whether to connect percentile points with lines.
        save_dir: Directory to save the plot. If None, uses csv_dir.
        filename: Output filename. If None, auto-generated.
        normalize: Whether to normalize the delay values by the max value in each CSV.
    """
    csv_dir = os.path.expanduser(str(csv_dir))
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"Directory not found: {csv_dir!r}")
    
    # Find all delay_summary CSVs
    pattern = os.path.join(csv_dir, "delay_summary_*.csv")
    all_files = sorted(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No delay_summary_*.csv files found in {csv_dir!r}")
    
    # Filter by world size if num_gpus is provided
    if num_gpus is not None:
        world_size_pattern = f"world_size{num_gpus}"
        files = [f for f in all_files if world_size_pattern in os.path.basename(f)]
        if not files:
            raise FileNotFoundError(
                f"No delay_summary CSVs found with '{world_size_pattern}' in filename. "
                f"Found {len(all_files)} total CSVs."
            )
    else:
        files = all_files
    
    print(f"📂 Found {len(files)} delay summary CSV(s) to process")
    
    # Extract topology from filename
    def extract_topology(filename: str) -> str:
        name = filename.replace("_delay.csv", "")
        known_topos = ["Fat_Tree", "Fat Tree", "Dragonfly+", "HyperX", "Rail_Only"]
        for topo in known_topos:
            if name.startswith(topo.replace(" ", "_")):
                return topo.replace("_", " ")
        parts = name.split("_")
        return parts[0].replace("_", " ")
    
    # Accumulate values per topology.
    # Structure: {topology: {percentile: [list of values]}}
    topo_values: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_percentiles = set()
    
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ⚠️ Skipping {os.path.basename(csv_path)}: {e}")
            continue
        
        if "file" not in df.columns:
            print(f"  ⚠️ Skipping {os.path.basename(csv_path)}: no 'file' column")
            continue
        
        # Find percentile columns
        pct_cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
        if not pct_cols:
            print(f"  ⚠️ Skipping {os.path.basename(csv_path)}: no percentile columns")
            continue
        
        percentiles = sorted([int(c[1:]) for c in pct_cols])
        all_percentiles.update(percentiles)
        
        max_val = None
        if normalize:
            # Find max value in this CSV for normalization (across all topologies and percentiles)
            max_val = 0.0
            for _, row in df.iterrows():
                for p in percentiles:
                    val = row.get(f"p{p}", 0)
                    if pd.notna(val) and val > max_val:
                        max_val = val

            if max_val <= 0:
                print(
                    f"  ⚠️ Skipping {os.path.basename(csv_path)}: "
                    "max value is 0 (required for normalization)"
                )
                continue

        # Accumulate normalized or raw values based on the flag.
        for _, row in df.iterrows():
            topo_name = extract_topology(row["file"])
            for p in percentiles:
                val = row.get(f"p{p}", 0)
                if pd.notna(val):
                    value = (val / max_val) if normalize else val
                    topo_values[topo_name][p].append(value)
    
    if not topo_values:
        raise RuntimeError("No valid data found in any CSV files")
    
    # Compute averages
    percentiles = sorted(all_percentiles)
    topo_averages: Dict[str, List[float]] = {}
    
    for topo_name, pct_dict in topo_values.items():
        avg_values = []
        for p in percentiles:
            vals = pct_dict.get(p, [])
            if vals:
                avg_values.append(np.mean(vals))
            else:
                avg_values.append(np.nan)
        topo_averages[topo_name] = avg_values
    
    # Generate title
    if title is None:
        title_prefix = "Average Normalized Delay Percentiles" if normalize else "Average Delay Percentiles"
        if num_gpus is not None:
            title = f"{title_prefix} ({num_gpus} GPUs, {len(files)} workloads)"
        else:
            title = f"{title_prefix} ({len(files)} workloads)"
    
    # Plot setup
    plt.figure(figsize=(14, 6))
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', 'D', 'x', 'v', 'p', '*', 'h', '+', '<', '>', '8', 'P', 'X']
    
    # Define specific order for topologies
    topo_order = ["Fat Tree", "HyperX", "Dragonfly+"]
    ordered_items = []
    for topo in topo_order:
        if topo in topo_averages:
            ordered_items.append((topo, topo_averages[topo]))
    # Add any remaining topologies not in the predefined order
    for topo, avg_values in topo_averages.items():
        if topo not in topo_order:
            ordered_items.append((topo, avg_values))
    
    for idx, (topo_name, avg_values) in enumerate(ordered_items):
        marker = markers[idx % len(markers)]
        color = default_colors[idx % len(default_colors)]

        # Keep full line, but only show markers at >=5 percentile distance.
        marker_indices = []
        last_marked_pct = None
        for i, p in enumerate(percentiles):
            if last_marked_pct is None or (p - last_marked_pct) >= 5:
                marker_indices.append(i)
                last_marked_pct = p
        # Always include final point for readability.
        if marker_indices and marker_indices[-1] != len(percentiles) - 1:
            marker_indices.append(len(percentiles) - 1)
        
        # Switched axes: delay (avg_values) on x, percentile on y
        if connect_points:
            plt.plot(avg_values, percentiles, color=color, label=topo_name,
                     linewidth=2, marker=marker, markersize=8, markevery=marker_indices)
        else:
            x_mark = [avg_values[i] for i in marker_indices if i < len(avg_values)]
            y_mark = [percentiles[i] for i in marker_indices if i < len(avg_values)]
            plt.scatter(x_mark, y_mark, color=color, label=topo_name,
                        marker=marker, s=80)
    
    plt.xlabel("Delay (ms)", fontsize=18)
    plt.ylabel("Percentile (%)", fontsize=18)
    plt.title(title, fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Show major ticks every 5% on the percentile axis (even if data has 1% spacing).
    ax = plt.gca()
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    plt.ylim(min(percentiles) - 5, max(percentiles) + 5)  # Low at bottom, high at top
    
    if use_log_y:
        plt.xscale("log")  # Log scale now on x-axis (delay)
    
    # Determine save location
    if save_dir is None:
        save_dir = csv_dir
    
    if filename is None:
        if num_gpus is not None:
            filename = f"avg_delay_percentiles_{num_gpus}gpus.png"
        else:
            filename = "avg_delay_percentiles_all.png"
    
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"📊 Saved average delay percentiles plot: {out_path}")
    
    # Also save the averaged data as CSV (using same topology order)
    csv_out_path = out_path.replace(".png", ".csv")
    rows = []
    for topo_name, avg_values in ordered_items:
        row = {"topology": topo_name}
        for p, v in zip(percentiles, avg_values):
            row[f"p{p}"] = v
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(csv_out_path, index=False)
    print(f"📄 Saved average delay percentiles CSV: {csv_out_path}")

    