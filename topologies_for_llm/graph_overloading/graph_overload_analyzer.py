from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt
import os

from .assign_and_plot import assign_od_to_edges_shortest_first, annotate_graph_with_loads, draw_fattree_overload
from .topologies.fat_tree import FatTree
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


def load_csv_to_csr(csv_path: Path, dtype=np.float32) -> sparse.csr_matrix:
    # Load dense (simple & robust), then sparsify.
    # For very large CSVs, consider chunked reading, but this is a good baseline.
    a = np.loadtxt(csv_path, delimiter=",", dtype=dtype)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"{csv_path} must be square n x n; got shape {a.shape}")

    # Convert to CSR sparse (stores only non-zeros)
    return sparse.csr_matrix(a)


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

    # Build topology for that world size
    ft = FatTree(num_nodes=n, switch_ports=switch_ports, down_ports=down_ports, link_capacity=1.0, link_weight=1.0)

    # Convert to NetworkX
    G = ft.convert_to_networkx()

    # Assign OD to edges (shortest-first)
    edge_load = assign_od_to_edges_shortest_first(G, OD, weight="weight")
    annotate_graph_with_loads(G, edge_load)

    # Summaries
    utils = [data.get("util", 0.0) for _, _, data in G.edges(data=True)]
    max_util = max(utils) if utils else 0.0
    overloaded = sum(1 for u, v, d in G.edges(data=True) if d.get("overloaded", False))

    print(f"\nWorkload: {workload_name}")
    print(f"Endpoints (GPUs): {n}")
    print(f"Graph nodes (incl switches): {G.number_of_nodes()}")
    print(f"Graph edges (directed): {G.number_of_edges()}")
    print(f"Max utilization: {max_util:.3f}")
    print(f"Overloaded edges: {overloaded}")

    # Draw (for large n, draw only top busy edges)
    draw_fattree_overload(G, max_edges_to_draw=3000, title=f"FatTree overload - {workload_name} (max util {max_util:.2f})")

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
        run_one_workload_on_fattree(workloads, chosen, switch_ports=64)


if __name__ == "__main__":
    main()