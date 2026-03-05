from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


class DragonflyPlus(Topology):
    """
    Dragonfly+ topology with leaf-spine groups and direct GPU-GPU links.

    Structure:
      - Multiple groups, each containing leaf and spine switches
      - Each group has (group_size / 2) leaf switches and (group_size / 2) spine switches
      - Number of groups = num_nodes / ((group_size / 2) * gpus_per_leaf)
      - GPUs within each leaf have direct links (full mesh)

    Intra-group connectivity:
      - Each leaf switch connects to `gpus_per_leaf` consecutive GPUs
      - Each leaf switch connects to ALL spine switches in its group

    Inter-group connectivity:
      - Spine switches connect to spine switches in other groups
      - Minimum connections: each spine connects to one spine per remote group

    Distance model:
      - Same leaf (consecutive gpus_per_leaf GPUs): 1 hop (direct GPU-GPU link)
      - Same group, different leaf: 4 hops (GPU → Leaf1 → Spine → Leaf2 → GPU, passes 3 switches)
      - Different groups: 5 hops (GPU → Leaf1 → Spine1 → Spine2 → Leaf2 → GPU, passes 4 switches)

    Parameters:
      num_nodes: Total number of GPUs/endpoints
      group_size: Total switches per group (leaf + spine), must be even
      gpus_per_leaf: GPUs per leaf switch (default 8)
      link_capacity: Capacity attribute for edges
      link_weight: Weight attribute for edges
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        router_ports: int = 64,           # Kept for API compatibility
        endpoints_per_router: int = 8,    # Alias for gpus_per_leaf
        group_size: int = 8,             # Total switches per group (leaf + spine)
        gpus_per_leaf: int | None = None, # GPUs per leaf switch
        inter_group_variant: str = "medium",  # Kept for API compatibility
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        # gpus_per_leaf defaults to endpoints_per_router for compatibility
        if gpus_per_leaf is None:
            gpus_per_leaf = endpoints_per_router

        if group_size < 2 or group_size % 2 != 0:
            raise ValueError("group_size must be an even number >= 2")
        if gpus_per_leaf < 1:
            raise ValueError("gpus_per_leaf must be >= 1")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.group_size = int(group_size)
        self.gpus_per_leaf = int(gpus_per_leaf)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)
        self.router_ports = int(router_ports)

        # Derived parameters
        self.leaves_per_group = self.group_size // 2
        self.spines_per_group = self.group_size // 2
        
        # Calculate number of groups
        gpus_per_group = self.leaves_per_group * self.gpus_per_leaf
        self.num_groups = int(np.ceil(num_nodes / gpus_per_group))
        
        # Total switches
        self.total_leaves = self.num_groups * self.leaves_per_group
        self.total_spines = self.num_groups * self.spines_per_group
        self.total_switches = self.total_leaves + self.total_spines

        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        gpus_per_leaf = self.gpus_per_leaf
        leaves_per_group = self.leaves_per_group
        spines_per_group = self.spines_per_group
        num_groups = self.num_groups

        # Node IDs layout:
        # - GPUs: [0, n-1]
        # - Leaf switches: [n, n + total_leaves - 1]
        # - Spine switches: [n + total_leaves, n + total_leaves + total_spines - 1]
        leaf_base = n
        spine_base = n + self.total_leaves
        total_nodes = n + self.total_switches

        def leaf_id(group: int, leaf_idx: int) -> int:
            """Get node ID for leaf switch."""
            return leaf_base + group * leaves_per_group + leaf_idx

        def spine_id(group: int, spine_idx: int) -> int:
            """Get node ID for spine switch."""
            return spine_base + group * spines_per_group + spine_idx

        def get_group_of_gpu(gpu: int) -> int:
            """Get the group index for a GPU."""
            gpus_per_group = leaves_per_group * gpus_per_leaf
            return gpu // gpus_per_group

        def get_leaf_of_gpu(gpu: int) -> Tuple[int, int]:
            """Get (group, leaf_idx) for a GPU."""
            gpus_per_group = leaves_per_group * gpus_per_leaf
            group = gpu // gpus_per_group
            local_gpu = gpu % gpus_per_group
            leaf_idx = local_gpu // gpus_per_leaf
            return group, leaf_idx

        # Build edges
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u)
            edges_v.append(v)
            edges_u.append(v)
            edges_v.append(u)

        # (1) GPU <-> Leaf connections
        # Each consecutive group of `gpus_per_leaf` GPUs connects to one leaf
        gpu_to_leaf = np.empty(n, dtype=np.int32)
        gpu_to_group = np.empty(n, dtype=np.int32)
        
        for gpu in range(n):
            group, local_leaf = get_leaf_of_gpu(gpu)
            leaf = leaf_id(group, local_leaf)
            gpu_to_leaf[gpu] = leaf
            gpu_to_group[gpu] = group
            add_bidir(gpu, leaf)

        # (1b) Direct GPU <-> GPU links within each leaf group (full mesh)
        # This creates direct distance-1 links between consecutive gpus_per_leaf GPUs
        total_leaf_groups = self.total_leaves
        for leaf_idx in range(total_leaf_groups):
            base_gpu = leaf_idx * gpus_per_leaf
            # Create full mesh within the leaf group
            for i in range(gpus_per_leaf):
                for j in range(i + 1, gpus_per_leaf):
                    gpu_i = base_gpu + i
                    gpu_j = base_gpu + j
                    if gpu_i < n and gpu_j < n:
                        add_bidir(gpu_i, gpu_j)

        # (2) Intra-group: Leaf <-> Spine connections (full bipartite within group)
        for g in range(num_groups):
            for li in range(leaves_per_group):
                for si in range(spines_per_group):
                    add_bidir(leaf_id(g, li), spine_id(g, si))

        # (3) Inter-group: Spine <-> Spine connections
        # Minimum connections: each spine connects to one spine per remote group
        # Using a round-robin assignment for balanced distribution
        if num_groups > 1:
            for g1 in range(num_groups):
                for g2 in range(g1 + 1, num_groups):
                    # Each spine in g1 connects to one spine in g2
                    # Use modular assignment for minimum connectivity
                    for si in range(spines_per_group):
                        # Connect spine si in g1 to spine si in g2
                        remote_spine = si % spines_per_group
                        add_bidir(spine_id(g1, si), spine_id(g2, remote_spine))

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )

        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "Dragonfly+ (leaf-spine groups)",
            "num_endpoints": n,
            "params": {
                "num_groups": num_groups,
                "group_size": self.group_size,
                "leaves_per_group": leaves_per_group,
                "spines_per_group": spines_per_group,
                "gpus_per_leaf": gpus_per_leaf,
                "router_ports": self.router_ports,
            },
            "distance_model": {
                "same_leaf": 1,
                "same_group_different_leaf": 4,
                "different_groups": 5,
            },
            "counts": {
                "total_nodes_including_switches": total_nodes,
                "total_leaves": self.total_leaves,
                "total_spines": self.total_spines,
                "total_switches": self.total_switches,
                "endpoints": n,
                "direct_gpu_links_per_leaf": gpus_per_leaf * (gpus_per_leaf - 1) // 2,
            },
            "node_ranges": {
                "endpoints": (0, n - 1),
                "leaves": (leaf_base, leaf_base + self.total_leaves - 1),
                "spines": (spine_base, spine_base + self.total_spines - 1),
            },
            "gpu_to_leaf": gpu_to_leaf,
            "gpu_to_group": gpu_to_group,
        }

        res = TopologyBuildResult(edges=edges, attrs={"capacity": capacity, "weight": weight}, meta=meta)
        self._cached_build = res
        return res

    def convert_to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX DiGraph.
        """
        res = self._cached_build if self._cached_build is not None else self.build_topology()
        edges = res.edges
        cap = res.attrs.get("capacity")
        w = res.attrs.get("weight")

        G = nx.DiGraph()

        total_nodes = res.meta.get("counts", {}).get("total_nodes_including_switches")
        if isinstance(total_nodes, int):
            G.add_nodes_from(range(total_nodes))

        for idx in range(edges.shape[0]):
            u, v = int(edges[idx, 0]), int(edges[idx, 1])
            attrs: Dict[str, Any] = {"edge_id": int(idx)}
            if cap is not None:
                attrs["capacity"] = float(cap[idx])
            if w is not None:
                attrs["weight"] = float(w[idx])
            G.add_edge(u, v, **attrs)

        G.graph["meta"] = res.meta
        return G
