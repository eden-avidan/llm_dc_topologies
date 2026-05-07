from __future__ import annotations

"""
Main experiment driver for graph-overloading studies.
Configures workloads/topologies, runs routing/delay analysis,
and orchestrates plot/CSV generation pipelines.
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import re
from typing import Tuple, List
from pathlib import Path
from copy import deepcopy
import numpy as np
from scipy import sparse


from assign_and_plot import (
    annotate_graph_with_loads,
    assign_od_to_edges_shortest,
    plot_edge_load_cdf_multiple,
    plot_shortest_path_heatmap,
    compute_gpu_to_gpu_delay_df,
    load_and_compare_delay_percentiles,
    plot_delay_percentiles_from_csv,
    get_edge_load_stats,
    save_effective_heatmap_csv,
    save_effective_heatmap_nonzero_distribution_csv,
    plot_average_cdf_from_csvs,
    get_graph_switch_and_link_counts,
    plot_average_delay_percentiles_from_dir,
)
from topologies.dragonfly_plus import DragonflyPlus
from topologies.fat_tree import FatTree
from topologies.hyperx import HyperX

matrices_moe = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices_moe'
matrices = '/Users/eavidan/Documents/topology_repo/simai/final_output/matrices'

this_dir = Path(__file__).parent

def workload_name_to_transport_csv_path(workload_name: str) -> str:
    name = workload_name[:-4] if workload_name.lower().endswith(".csv") else workload_name
    csv_path = os.path.join(matrices, f"{name}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Transport CSV for workload {workload_name!r} not found at {csv_path!r}. "
            f"Expected transport CSVs under {matrices!r}."
        )
    return csv_path

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
    


def create_all_topologies_and_graphs(
    workloads: dict,
    workload_name: str,
    *,
    workload_type: str,          # "moe" or "dense"
    root_save_dir: str,          # path to edge_load_comparisons
    switch_ports: int = 128,
    router_ports: int = 256,
    endpoints_per_router: int = 8,
    inter_group_variant: str = "medium",
    use_log_x: bool = True,
    include_zeros: bool = True,
    fat_tree_base_graph: nx.Graph | None = None,
    hyperx_base_graph: nx.Graph | None = None,
    dragonfly_plus_base_graph: nx.Graph | None = None,
):
    OD = workloads[workload_name]
    n = OD.shape[0]

    fat_tree_graph = fat_tree_base_graph
    if fat_tree_graph is None:
        fat_tree_graph = FatTree(
            num_nodes=n,
            switch_ports=switch_ports,
            node_size=8,
            link_capacity=1.0,
            link_weight=1.0,
        ).convert_to_networkx()

    hyperx_graph = hyperx_base_graph
    if hyperx_graph is None:
        hyperx_graph = HyperX(
            num_nodes=n,
            router_ports=router_ports,
            endpoints_per_router=8,
            aggregate_tp_groups=False,
            transport_csv_path=workload_name_to_transport_csv_path(workload_name=workload_name),
            link_capacity=1.0,
            link_weight=1.0,
        ).convert_to_networkx()

    dragonfly_plus_graph = dragonfly_plus_base_graph
    if dragonfly_plus_graph is None:
        dragonfly_plus_graph = DragonflyPlus(
            num_nodes=n,
            router_ports=router_ports,
            gpus_per_leaf=8,
            inter_group_variant=inter_group_variant,
            link_capacity=1.0,
            link_weight=1.0,
        ).convert_to_networkx()

    base_graphs = {
        "Fat Tree": fat_tree_graph,
        "HyperX": hyperx_graph,
        "Dragonfly+": dragonfly_plus_graph,
    }

    for graph in base_graphs.values():
        switch_counts = get_graph_switch_and_link_counts(graph)
        print(f"Counts for {graph.graph['meta']['topology']}: {switch_counts}")

    variants = [
        # ("single_path", False),
        ("equal_spread", True),
    ]

    for variant_dirname, split_equal_shortest in variants:
        graphs_variant = {}
        G_dict = {}
        edge_loads = {}
        save_dir = str(Path(root_save_dir) / workload_type / variant_dirname)
        delay_save_dir = str(Path(save_dir) / "delay_matrices")
        os.makedirs(delay_save_dir, exist_ok=True)
        max_capacity = 3.5e13
        
        for topo_name, G_base in base_graphs.items():
            G = deepcopy(G_base)
            edge_load = assign_od_to_edges_shortest(
                G, OD, weight="weight", split_equal_shortest=split_equal_shortest, num_endpoints=n
            )

            annotate_graph_with_loads(G, edge_load, capacity=None)  # First pass to get loads
            stats = get_edge_load_stats(G)
            capacity = max(max_capacity, stats['max'] * 1.5)  # 50% headroom above max load
            max_capacity = capacity
            graphs_variant[topo_name] = G
            edge_loads[topo_name] = edge_load
            G_dict[topo_name] = G
        print(f"Using capacity={max_capacity:.2e}")

        
        
        for topo_name, G_base in base_graphs.items():
            G = G_dict[topo_name]
            edge_load = edge_loads[topo_name]
            annotate_graph_with_loads(G, edge_load, capacity=max_capacity)  # Re-annotate with capacity
            graphs_variant[topo_name] = G

            df_delay = compute_gpu_to_gpu_delay_df(
                G,
                OD,
                num_endpoints=n,
                split_equal_shortest=False,           # match your variant
                bandwidth_bytes_per_sec=50e9,        # example: 50 GB/s (set your own)
                alpha_per_hop_sec=0.0,               # optional
                save_dir=delay_save_dir,
                filename=f"{topo_name.replace(' ', '_')}_{workload_name}_delay.csv",
            )
        
        cdf_dir = str(Path(save_dir) / "cdf")
        histogram_dir = str(Path(save_dir) / "histogram")
        percentiles_dir = str(Path(save_dir) / "percentiles")

        
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

        percentiles_df = load_and_compare_delay_percentiles(
            delay_save_dir,
            workload_name=workload_name,
        )
        
        percentiles_df.to_csv(os.path.join(delay_save_dir, f"percentiles_{workload_name}.csv"))
        
        plot_delay_percentiles_from_csv(
            os.path.join(delay_save_dir, f"percentiles_{workload_name}.csv"),
            title=f"Delay percentiles - {workload_name} ({variant_dirname})",
            save_dir=delay_save_dir,
            filename=os.path.join(percentiles_dir, f"percentiles_{workload_name}.png"),
        )


def run_effective_heatmap_analysis(workload_type: str, should_exit: bool = False):
        save_effective_heatmap_csv(
            heatmaps_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "heatmaps"),
            transport_dir="/Users/eavidan/Documents/topology_repo/simai/final_output/matrices",
            save_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "effective_heatmaps"),
        )

        save_effective_heatmap_nonzero_distribution_csv(
            effective_heatmaps_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "effective_heatmaps"),
            save_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "effective_heatmaps_nonzero_distribution"),
        )
        if should_exit:
            sys.exit(0)

def run_graph_overload_analysis(workload_type: str, should_exit: bool = False):
    plot_average_cdf_from_csvs(
        csv_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "cdf"),
        world_size=1024,
        title=f"Average Edge-load CDF (1024 GPUs)",
        use_log_x=True,
        save_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "cdf"),
        filename=f"average_cdf_1024gpus.png",
    )

    plot_average_delay_percentiles_from_dir(
        csv_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "delay_matrices"),
        num_gpus=1024,
        title=f"Average Delay Percentiles (1024 GPUs)",
        use_log_y=True,
        connect_points=True,
        save_dir=os.path.join(this_dir, "edge_load_comparisons", workload_type, "equal_spread", "delay_matrices"),
        filename=f"average_delay_percentiles_1024gpus.png",
        normalize=False,
    )
    if should_exit:
        sys.exit(0)

def main() -> None:
    matrices_dirs = [
        matrices,
        # matrices_moe,
    ]
    workload_types = [
        "dense",
        # "moe",
    ]

    for matrices_dir, workload_type in zip(matrices_dirs, workload_types):
        run_effective_heatmap_analysis(workload_type=workload_type)
        run_graph_overload_analysis(workload_type=workload_type, should_exit=True)
        
        print(f"\n=== Loading {workload_type} from {matrices_dir} ===")
        workloads = load_workloads_from_dir(Path(matrices_dir))
        workloads = {
            name: M
            for name, M in workloads.items()
            if ("tp1-" not in name and "tp2-" not in name and "tp4-" not in name)
        }

        # quick summary (optional)
        for name, M in list(workloads.items())[:5]:
            n = M.shape[0]
            nnz = M.nnz
            density = nnz / (n * n)
            total = float(M.sum())
            print(f"{name}: n={n}, nnz={nnz}, density={density:.6f}, total_bytes={total:.3e}")
        if len(workloads) > 5:
            print(f"... ({len(workloads)-5} more)")

        world_sizes = [1024]
        for world_size in world_sizes:
            # Find ALL workloads that match this world size
            matching_workloads = [name for name, M in workloads.items() if M.shape[0] == world_size]
            
            if not matching_workloads:
                print(f"No workload with world_size={world_size} found, skipping.")
                continue
            
            print(f"Found {len(matching_workloads)} workloads with world_size={world_size}")
            base_graphs = {
                "Fat Tree": FatTree(
                    num_nodes=n,
                    switch_ports=128,
                    node_size=8,
                    link_capacity=1.0,
                    link_weight=1.0,
                ).convert_to_networkx(),
                "Dragonfly+": DragonflyPlus(
                    num_nodes=n,
                    router_ports=256,
                    gpus_per_leaf=8,
                    inter_group_variant="medium",
                    link_capacity=1.0,
                    link_weight=1.0
                ).convert_to_networkx()

            }
            for chosen in matching_workloads:
                create_all_topologies_and_graphs(
                    workloads,
                    chosen,
                    workload_type=workload_type,
                    root_save_dir=os.path.join(this_dir, "edge_load_comparisons"),
                    endpoints_per_router=8,
                    inter_group_variant="medium",
                    fat_tree_base_graph=base_graphs["Fat Tree"],
                    dragonfly_plus_base_graph=base_graphs["Dragonfly+"],
                )

if __name__ == "__main__":
    main()