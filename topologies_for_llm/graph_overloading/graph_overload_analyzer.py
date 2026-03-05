from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path
from copy import deepcopy

from assign_and_plot import (
    annotate_graph_with_loads,
    assign_od_to_edges_shortest,
    plot_edge_load_cdf,
    plot_edge_load_cdf_multiple,
    plot_edge_load_bucket_hist_multiple,
    plot_edge_load_percentiles_multiple,
    plot_shortest_path_heatmap
)
from topologies.dragonfly_plus import DragonflyPlus
from topologies.fat_tree import FatTree
from topologies.hyperx import HyperX

matrices_moe = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices_moe'
matrices = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices'

this_dir = Path(__file__).parent

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

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight")

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

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight")

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


def run_one_workload_on_dragonfly_plus(workloads: dict, workload_name: str, *, router_ports=64, inter_group_variant="medium"):
    OD = workloads[workload_name]
    n = OD.shape[0]

    # Dragonfly+ paper-faithful implementation (Shpiner et al., IEEE 2017):
    # - Enforces p=l=s=h=k/2 balancing rule (Equation 1)
    # - Global links ONLY on spine routers (spine-to-spine)
    # - Three inter-group variants: "largest" (minimal), "medium", "small" (parallel links)
    dfp = DragonflyPlus(
        num_nodes=n,
        router_ports=router_ports,
        inter_group_variant=inter_group_variant,
        link_capacity=1.0,
        link_weight=1.0,
    )
    G = dfp.convert_to_networkx()

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight")

    # Infinite capacity => util is irrelevant; just store load
    annotate_graph_with_loads(G, edge_load, capacity=None)

    print(f"\nWorkload: {workload_name}")
    print(f"Endpoints (GPUs): {n}")
    print(f"Graph nodes (incl switches): {G.number_of_nodes()}")
    print(f"Graph edges (directed): {G.number_of_edges()}")

    plot_edge_load_cdf(
        G,
        title=f"Dragonfly+ edge-load CDF - {workload_name}",
        use_log_x=True,
        include_zeros=True,
    )

def create_all_topologies_and_graphs(
    workloads: dict,
    workload_name: str,
    *,
    workload_type: str,          # "moe" or "dense"
    root_save_dir: str,          # path to edge_load_comparisons
    switch_ports: int = 64,
    down_ports: int | None = None,
    router_ports: int = 64,
    endpoints_per_router: int = 8,
    inter_group_variant: str = "medium",
    use_log_x: bool = True,
    include_zeros: bool = True,
):
    OD = workloads[workload_name]
    n = OD.shape[0]

    base_graphs = {
        "Fat Tree": FatTree(num_nodes=n).convert_to_networkx(),
        "HyperX": HyperX(
            num_nodes=n,
            router_ports=router_ports,
            endpoints_per_router=8,
            # dims computed automatically based on num_nodes and endpoints_per_router
            link_capacity=1.0,
            link_weight=1.0,
        ).convert_to_networkx(),
        # "HyperX_1": HyperX(
        #     num_nodes=n,
        #     router_ports=router_ports,
        #     endpoints_per_router=1,
        #     # dims computed automatically based on num_nodes and endpoints_per_router
        #     link_capacity=1.0,
        #     link_weight=1.0,
        # ).convert_to_networkx(),
        "Dragonfly+": DragonflyPlus(
            num_nodes=n,
            router_ports=router_ports,
            gpus_per_leaf=8,
            inter_group_variant=inter_group_variant,
            link_capacity=1.0,
            link_weight=1.0
        ).convert_to_networkx()#,
        # "DragonflyPlus_1": DragonflyPlus(
        #     num_nodes=n,
        #     router_ports=router_ports,
        #     gpus_per_leaf=1,
        #     inter_group_variant=inter_group_variant,
        #     link_capacity=1.0,
        #     link_weight=1.0
        # ).convert_to_networkx()
    }

    variants = [
        ("single_path", False),
        ("equal_spread", True),
    ]

    for variant_dirname, split_equal_shortest in variants:
        graphs_variant = {}

        for topo_name, G_base in base_graphs.items():
            G = deepcopy(G_base)
            edge_load = assign_od_to_edges_shortest(
                G, OD, weight="weight", split_equal_shortest=split_equal_shortest
            )
            annotate_graph_with_loads(G, edge_load, capacity=None)
            graphs_variant[topo_name] = G

        heatmap_save_dir = str(Path(root_save_dir) / workload_type / variant_dirname / "heatmaps")
        for topo_name, graph in graphs_variant.items():
            plot_shortest_path_heatmap(
                graph,
                num_endpoints=n,  # Only show GPU nodes (0 to n-1), matching transport matrix
                title=f"{topo_name} latency heatmap (world size={n})",
                x_label="Target GPU",
                y_label="Source GPU",
                save_dir=heatmap_save_dir,
                filename=f"{topo_name.replace(' ', '_')}_{workload_name}.png",
            )
        save_dir = str(Path(root_save_dir) / workload_type / variant_dirname)
        cdf_dir = str(Path(save_dir) / "cdf")
        histogram_dir = str(Path(save_dir) / "histogram")
        percentiles_dir = str(Path(save_dir) / "percentiles")
        
        # Create directories if they don't exist
        os.makedirs(cdf_dir, exist_ok=True)
        os.makedirs(histogram_dir, exist_ok=True)
        os.makedirs(percentiles_dir, exist_ok=True)
        
        plot_edge_load_cdf_multiple(
            graphs_variant,
            title=f"Edge-load CDF - {workload_name} ({variant_dirname})",
            use_log_x=use_log_x,
            save_dir=save_dir,
            include_zeros=include_zeros,
            filename=os.path.join(cdf_dir, f"cdf_{workload_name}.png"),
        )

        plot_edge_load_bucket_hist_multiple(
            graphs_variant,
            title=f"Edge-load bucket histogram - {workload_name} ({variant_dirname})",
            include_zeros=include_zeros,
            save_dir=save_dir,
            filename=os.path.join(histogram_dir, f"histogram_{workload_name}.png"),
        )

        plot_edge_load_percentiles_multiple(
            graphs_variant,
            title=f"Edge-load percentiles - {workload_name} ({variant_dirname})",
            include_zeros=include_zeros,
            save_dir=save_dir,
            filename=os.path.join(percentiles_dir, f"percentiles_{workload_name}.png"),
        )


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

        # base_save_path = str(os.path.join(this_dir, "edge_load_comparisons", workload_type))
        # total_workloads = len(workloads)
        # for i, workload_name in enumerate(workloads.keys()):
        #     print(f"\nProcessing workload: {workload_name}")
        #     print(f"Progress: {(i+1) / total_workloads * 100:.1f}%")
        #     create_all_topologies_and_graphs(
        #         workloads,
        #         workload_name,
        #         workload_type=workload_type,
        #         root_save_dir=os.path.join(this_dir, "edge_load_comparisons"),
        #         switch_ports=128,
        #         down_ports=None,
        #         router_ports=128,
        #         endpoints_per_router=8,
        #         inter_group_variant="medium",
        #     )
        world_sizes = [128, 1024]
        for i in world_sizes:
            chosen = next((name for name, M in workloads.items() if M.shape[0] == i), None)
            if chosen is None:
                print("No workload with world_size=128 found, using first available.")
                chosen = next(iter(workloads.keys()))

            create_all_topologies_and_graphs(
                workloads,
                chosen,
                workload_type=workload_type,
                root_save_dir=os.path.join(this_dir, "edge_load_comparisons"),
                switch_ports=128,
                down_ports=None,
                router_ports=128,
                endpoints_per_router=8,
                inter_group_variant="medium",
            )

if __name__ == "__main__":
    main()