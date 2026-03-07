from __future__ import annotations

"""
Helper entry points for running a single workload on one topology at a time
(Fat Tree / HyperX / Dragonfly+) and plotting quick CDF diagnostics.
"""

from assign_and_plot import (
    annotate_graph_with_loads,
    assign_od_to_edges_shortest,
    plot_edge_load_cdf,
)
from topologies.dragonfly_plus import DragonflyPlus
from topologies.fat_tree import FatTree
from topologies.hyperx import HyperX


def run_one_workload_on_fattree(workloads: dict, workload_name: str, *, switch_ports=64, down_ports=None, num_gpus=0):
    OD = workloads[workload_name]
    n = OD.shape[0]

    ft = FatTree(num_nodes=n, switch_ports=switch_ports, down_ports=down_ports, link_capacity=1.0, link_weight=1.0)
    G = ft.convert_to_networkx()

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight", num_endpoints=n)

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


def run_one_workload_on_hyperx(workloads: dict, workload_name: str, *, router_ports=64, endpoints_per_router=8, num_gpus=0):
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

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight", num_endpoints=n)

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


def run_one_workload_on_dragonfly_plus(workloads: dict, workload_name: str, *, router_ports=64, inter_group_variant="medium", num_gpus=0):
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

    edge_load = assign_od_to_edges_shortest(G, OD, weight="weight", num_endpoints=n)

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