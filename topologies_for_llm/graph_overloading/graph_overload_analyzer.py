from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

from assign_and_plot import (
    annotate_graph_with_loads,
    assign_od_to_edges_shortest_first,
    draw_fattree_load_heat,
    draw_fattree_overload,
    plot_edge_load_cdf,
    plot_edge_load_cdf_multiple
)
from topologies.dragonfly_plus import DragonflyPlus
from topologies.fat_tree import FatTree
from topologies.hyperx import HyperX

matrices_moe = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices_moe'
matrices = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices'

#!/usr/bin/env python3
"""
Load multiple CSV transport matrices into a dict:
  workloads[workload_name] -> scipy.sparse.csr_matrix

Assumptions:
- Each CSV is an n x n numeric matrix (no header row/col).
- Values are transported bytes from i -> j.
- Zeros are common (sparse-friendly).
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import sparse


def load_csv_to_csr(csv_path: Path, dtype=np.float32, return_labels: bool = False):
    """
    Loads a transport matrix CSV into a CSR sparse matrix.

    Supports BOTH formats:
      1) Pure numeric n×n CSV (no headers)
      2) Labeled CSV with header row + first column as row labels, like:
           GPU,GPU0,GPU1,...
           GPU0,0,123,...
           GPU1,...

    If return_labels=True, returns (csr_matrix, labels_list).
    Otherwise returns csr_matrix.
    """
    try:
        # Fast path for pure numeric CSV
        a = np.loadtxt(csv_path, delimiter=",", dtype=dtype)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"{csv_path} must be square n×n; got shape {a.shape}")
        M = sparse.csr_matrix(a)
        return (M, list(range(a.shape[0]))) if return_labels else M

    except ValueError:
        # Fallback: labeled CSV
        df = pd.read_csv(csv_path, index_col=0)

        # Strip whitespace on labels
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()

        # If columns contain the same labels as rows, align them to the same order
        if set(df.columns) == set(df.index):
            df = df.loc[df.index, df.index]

        # Convert all to numeric (coerce non-numeric to NaN -> fill with 0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Validate square
        if df.shape[0] != df.shape[1]:
            raise ValueError(f"{csv_path} must be square after parsing; got shape {df.shape}")

        a = df.to_numpy(dtype=dtype, copy=False)
        M = sparse.csr_matrix(a)

        labels = df.index.tolist()
        return (M, labels) if return_labels else M


def load_workloads_from_dir(matrices_dir: Path, pattern: str = "*.csv", dtype=np.float32):
    workloads = {}
    for csv_path in sorted(matrices_dir.glob(pattern)):
        name = csv_path.stem
        if name in workloads:
            raise ValueError(f"Duplicate workload name: {name}")
        workloads[name] = load_csv_to_csr(csv_path, dtype=dtype)
    print(f"Total {len(workloads)} workloads from {matrices_dir}")
    return workloads
    
def run_one_workload_on_fattree(workloads: dict, workload_name: str, *, switch_ports=64, down_ports=None):
    OD = workloads[workload_name]
    n = OD.shape[0]

    ft = FatTree(num_nodes=n, switch_ports=switch_ports, down_ports=down_ports, link_capacity=1.0, link_weight=1.0)
    G = ft.convert_to_networkx()

    edge_load = assign_od_to_edges_shortest_first(G, OD, weight="weight")

    # Infinite capacity => util is irrelevant; just store load
    annotate_graph_with_loads(G, edge_load, capacity=None)

    print(f"\nWorkload: {workload_name}")
    print(f"Endpoints (GPUs): {n}")
    print(f"Graph nodes (incl switches): {G.number_of_nodes()}")
    print(f"Graph edges (directed): {G.number_of_edges()}")

    plot_edge_load_cdf(
        G,
        title=f"FatTree edge-load CDF - {workload_name}",
        use_log_x=True,
        include_zeros=True,
    )


def run_one_workload_on_hyperx(workloads: dict, workload_name: str, *, router_ports=64, endpoints_per_router=8):
    OD = workloads[workload_name]
    n = OD.shape[0]

    hx = HyperX(
        num_nodes=n,
        router_ports=router_ports,
        endpoints_per_router=endpoints_per_router,
        link_capacity=1.0,
        link_weight=1.0,
    )
    G = hx.convert_to_networkx()

    edge_load = assign_od_to_edges_shortest_first(G, OD, weight="weight")

    # Infinite capacity => util is irrelevant; just store load
    annotate_graph_with_loads(G, edge_load, capacity=None)

    print(f"\nWorkload: {workload_name}")
    print(f"Endpoints (GPUs): {n}")
    print(f"Graph nodes (incl switches): {G.number_of_nodes()}")
    print(f"Graph edges (directed): {G.number_of_edges()}")

    plot_edge_load_cdf(
        G,
        title=f"HyperX edge-load CDF - {workload_name}",
        use_log_x=True,
        include_zeros=True,
    )


def run_one_workload_on_dragonfly_plus(workloads: dict, workload_name: str, *, router_ports=64, endpoints_per_router=8, global_links_per_router=8):
    OD = workloads[workload_name]
    n = OD.shape[0]

    dfp = DragonflyPlus(
        num_nodes=n,
        router_ports=router_ports,
        endpoints_per_router=endpoints_per_router,
        global_links_per_router=global_links_per_router,
        link_capacity=1.0,
        link_weight=1.0,
    )
    G = dfp.convert_to_networkx()

    edge_load = assign_od_to_edges_shortest_first(G, OD, weight="weight")

    # Infinite capacity => util is irrelevant; just store load
    annotate_graph_with_loads(G, edge_load, capacity=None)

    print(f"\nWorkload: {workload_name}")
    print(f"Endpoints (GPUs): {n}")
    print(f"Graph nodes (incl switches): {G.number_of_nodes()}")
    print(f"Graph edges (directed): {G.number_of_edges()}")

    plot_edge_load_cdf(
        G,
        title=f"DragonflyPlus edge-load CDF - {workload_name}",
        use_log_x=True,
        include_zeros=True,
    )

def create_all_topologies_and_graphs(workloads: dict, workload_name: str, switch_ports: int = 64, down_ports: int = None, router_ports: int = 64, endpoints_per_router: int = 8, global_links_per_router: int = 8):
    graphs = {}
    OD = workloads[workload_name]
    n = OD.shape[0]

    graphs["Fat Tree"] = FatTree(num_nodes=n, switch_ports=switch_ports, down_ports=down_ports, link_capacity=1.0, link_weight=1.0).convert_to_networkx()
    graphs["HyperX"] = HyperX(num_nodes=n, router_ports=router_ports, endpoints_per_router=endpoints_per_router, link_capacity=1.0, link_weight=1.0).convert_to_networkx()
    graphs["Dragonfly+"] = DragonflyPlus(num_nodes=n, router_ports=router_ports, endpoints_per_router=endpoints_per_router, global_links_per_router=global_links_per_router, link_capacity=1.0, link_weight=1.0).convert_to_networkx()
    
    for G in graphs.values():
        edge_load = assign_od_to_edges_shortest_first(G, OD, weight="weight")
        annotate_graph_with_loads(G, edge_load, capacity=None)
    
    plot_edge_load_cdf_multiple(graphs, title=f"Edge-load CDF - {workload_name}", use_log_x=True, include_zeros=True)

def main() -> None:
    matrices_dirs = [matrices_moe, matrices]
    workload_types = ["moe", "dense"]

    for matrices_dir, workload_type in zip(matrices_dirs, workload_types):
        print(f"\n=== Loading {workload_type} from {matrices_dir} ===")
        workloads = load_workloads_from_dir(Path(matrices_dir))

        # quick summary (optional)
        for name, M in list(workloads.items())[:5]:
            n = M.shape[0]
            nnz = M.nnz
            density = nnz / (n * n)
            total = float(M.sum())
            print(f"{name}: n={n}, nnz={nnz}, density={density:.6f}, total_bytes={total:.3e}")
        if len(workloads) > 5:
            print(f"... ({len(workloads)-5} more)")

        chosen = next(iter(workloads.keys()))
        # run_one_workload_on_fattree(workloads, chosen, switch_ports=64)
        # run_one_workload_on_hyperx(workloads, chosen, router_ports=64, endpoints_per_router=8)
        # run_one_workload_on_dragonfly_plus(workloads, chosen, router_ports=64, endpoints_per_router=8, global_links_per_router=8)

        create_all_topologies_and_graphs(workloads, chosen, switch_ports=64, down_ports=None, router_ports=64, endpoints_per_router=8, global_links_per_router=8)
if __name__ == "__main__":
    main()