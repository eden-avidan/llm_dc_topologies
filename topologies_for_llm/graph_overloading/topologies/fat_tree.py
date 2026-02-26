from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult

class FatTree(Topology):
    """
    Port-limited 3-stage FatTree/Clos-like topology built from one switch type.
    (This is the same model as your last FatTree, with an added convert_to_networkx()).
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        switch_ports: int = 64,
        down_ports: int = 32,  # shared for L1 and L2; default = floor(P/2)
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if not isinstance(switch_ports, int) or switch_ports <= 1:
            raise ValueError(f"switch_ports must be an int > 1, got {switch_ports!r}")
        self.switch_ports = switch_ports

        d = down_ports if down_ports is not None else switch_ports // 2
        if not isinstance(d, int) or d <= 0 or d >= switch_ports:
            raise ValueError("down_ports must be an int in [1, switch_ports-1]")
        self.down_ports = d

        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        # Optional cache so convert_to_networkx() doesn't rebuild repeatedly
        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n_gpus = self.num_nodes
        P = self.switch_ports

        # Shared split
        l1_down = self.down_ports
        l1_up = P - l1_down
        l2_down = self.down_ports
        l2_up = P - l2_down

        if l1_up <= 0 or l2_up <= 0:
            raise ValueError("Invalid split: no uplinks available (down_ports too large).")

        # 1) Minimum L1
        num_l1 = int(np.ceil(n_gpus / l1_down))

        # 2) Pods
        leaves_per_pod_cap = l2_down
        num_pods = int(np.ceil(num_l1 / leaves_per_pod_cap))

        # 3) L2 per pod
        l2_per_pod = l1_up
        total_l2 = num_pods * l2_per_pod

        # 4) Minimum L3
        total_l2_uplinks = total_l2 * l2_up
        num_l3 = int(np.ceil(total_l2_uplinks / P))

        # 5) IDs
        next_id = n_gpus
        l1_ids = np.arange(next_id, next_id + num_l1, dtype=np.int32)
        next_id += num_l1

        l2_ids_by_pod: list[np.ndarray] = []
        for _ in range(num_pods):
            ids = np.arange(next_id, next_id + l2_per_pod, dtype=np.int32)
            l2_ids_by_pod.append(ids)
            next_id += l2_per_pod

        l3_ids = np.arange(next_id, next_id + num_l3, dtype=np.int32)
        next_id += num_l3
        total_nodes = int(next_id)

        # 6) GPU -> L1
        gpu_to_l1 = np.empty(n_gpus, dtype=np.int32)
        for g in range(n_gpus):
            gpu_to_l1[g] = l1_ids[g // l1_down]

        # L1 blocks per pod
        l1_to_pod: Dict[int, int] = {}
        l1_blocks: list[np.ndarray] = []
        start = 0
        for pod in range(num_pods):
            block = l1_ids[start : start + leaves_per_pod_cap]
            l1_blocks.append(block)
            for lid in block.tolist():
                l1_to_pod[int(lid)] = pod
            start += leaves_per_pod_cap

        # 7) Edges (bidirectional physical links => 2 directed edges)
        edges_u: list[int] = []
        edges_v: list[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (a) GPU <-> L1
        for g in range(n_gpus):
            add_bidir(int(g), int(gpu_to_l1[g]))

        # (b) L1 <-> L2 within pod
        for pod, l1_block in enumerate(l1_blocks):
            l2_block = l2_ids_by_pod[pod]
            for lid in l1_block.tolist():
                for sid in l2_block.tolist():
                    add_bidir(int(lid), int(sid))

        # (c) L2 <-> L3 round-robin, using all L2 uplink ports
        l3_list = l3_ids.tolist()
        l3_idx = 0
        for pod in range(num_pods):
            for sid in l2_ids_by_pod[pod].tolist():
                for _ in range(l2_up):
                    uid = l3_list[l3_idx % len(l3_list)]
                    l3_idx += 1
                    add_bidir(int(sid), int(uid))

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )

        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "FatTree",
            "num_gpus": n_gpus,
            "switch_ports": P,
            "shared_split": {"down_ports": self.down_ports, "up_ports": P - self.down_ports},
            "counts": {
                "L1_leaves": num_l1,
                "pods": num_pods,
                "L2_spines_total": total_l2,
                "L2_spines_per_pod": l2_per_pod,
                "L3_uppers": num_l3,
                "total_nodes_including_switches": total_nodes,
            },
            "gpu_to_l1": gpu_to_l1,
            "l1_to_pod": l1_to_pod,
            "l2_ids_by_pod": l2_ids_by_pod,
            "l3_ids": l3_ids,
        }

        res = TopologyBuildResult(edges=edges, attrs={"capacity": capacity, "weight": weight}, meta=meta)
        self._cached_build = res
        return res

    def convert_to_networkx(self) -> nx.DiGraph:
        """
        Builds (or reuses cached) TopologyBuildResult and converts it to a NetworkX DiGraph.

        Determinism: edges are added in sorted (u,v) order so that
        tie-breaking that depends on adjacency iteration order is stable.
        """
        res = self._cached_build if self._cached_build is not None else self.build_topology()

        edges = res.edges
        cap = res.attrs.get("capacity")
        w = res.attrs.get("weight")

        # Sort edges by (u, v) for deterministic neighbor iteration order
        order = np.lexsort((edges[:, 1], edges[:, 0]))

        G = nx.DiGraph()
        # Add nodes explicitly (optional; NetworkX will add on edge insert anyway)
        total_nodes = res.meta.get("counts", {}).get("total_nodes_including_switches", None)
        if isinstance(total_nodes, int):
            G.add_nodes_from(range(total_nodes))

        for idx in order:
            u, v = int(edges[idx, 0]), int(edges[idx, 1])
            attrs: Dict[str, Any] = {"edge_id": int(idx)}
            if cap is not None:
                attrs["capacity"] = float(cap[idx])
            if w is not None:
                attrs["weight"] = float(w[idx])
            G.add_edge(u, v, **attrs)

        # Keep meta around for debugging / reporting
        G.graph["meta"] = res.meta
        return G