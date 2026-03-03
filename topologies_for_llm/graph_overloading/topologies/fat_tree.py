from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class FatTree(Topology):
    """
    Radix-128 fabric with explicit 8-GPU HBI hubs AND explicit leaf-class hubs.

    Requirements implemented:
      - HBI distance is 2: GPU_a -> HBI -> GPU_b (same HBI)
      - Leaf-class distance is 2: GPU_i -> Leaf(offset=i%8) -> GPU_{i+8k} (within same pod)
      - Leaves connect through spines (within pod): GPU -> Leaf -> Spine -> Leaf -> GPU (<=4 in one pod)
      - Spines connect through super-spines when multiple pods exist (cross-pod paths become >=6)

    Notes:
      - This is a port-limited model for switches (leaf/spine/super): each has <=128 incident links.
      - HBI hubs are modeled as abstract hubs and NOT port-limited (easy to add if you want).
      - For small sizes (e.g., 128 GPUs), we build ONE pod; super-spines are not used.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        switch_ports: int = 128,
        hbi_size: int = 8,
        leaves_per_pod: int = 8,   # fixed by your stride-8 leaf rule
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
        leaf_uplinks: int = 16,    # default; must satisfy leaf_down + leaf_uplinks <= 128
    ) -> None:
        super().__init__(num_nodes)

        if switch_ports != 128:
            raise ValueError("This topology is specifically for radix-128 switches.")
        if hbi_size != 8:
            raise ValueError("This topology is specifically for HBI groups of 8.")
        if leaves_per_pod != 8:
            raise ValueError("This topology implements the stride-8 leaf rule, so leaves_per_pod must be 8.")
        if self.num_nodes % hbi_size != 0:
            raise ValueError("num_nodes must be divisible by 8 (HBI size).")
        if not _is_power_of_two(self.num_nodes):
            raise ValueError("num_nodes (world size) must be a power of two.")

        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        if not isinstance(leaf_uplinks, int) or leaf_uplinks <= 0 or leaf_uplinks >= switch_ports:
            raise ValueError("leaf_uplinks must be an int in [1, switch_ports-1].")

        self.switch_ports = int(switch_ports)
        self.hbi_size = int(hbi_size)
        self.leaves_per_pod = int(leaves_per_pod)
        self.leaf_uplinks = int(leaf_uplinks)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n_gpus = self.num_nodes
        P = self.switch_ports
        H = self.hbi_size
        L = self.leaves_per_pod  # = 8

        # ---- Pod sizing with the stride-8 leaf rule ----
        # In a pod, leaf(offset i) must connect all GPUs with local index == i mod 8.
        # So per-leaf downlinks = pod_size / 8.
        # Leaf also needs uplinks to spines; enforce radix 128:
        #   (pod_size/8) + leaf_uplinks <= 128
        max_leaf_down = P - self.leaf_uplinks
        max_pod_size = 8 * max_leaf_down

        # Choose largest power-of-two pod_size <= max_pod_size, but not exceeding n_gpus.
        # This keeps pods uniform and preserves your leaf rule inside each pod.
        pod_size = 1
        while (pod_size * 2) <= max_pod_size and (pod_size * 2) <= n_gpus:
            pod_size *= 2
        if pod_size < 8:
            raise ValueError("Pod size ended up too small; check leaf_uplinks vs radix constraint.")
        if pod_size % 8 != 0:
            # should not happen given power-of-two and >=8
            raise ValueError("pod_size must be divisible by 8.")

        num_pods = int(np.ceil(n_gpus / pod_size))

        leaf_down = pod_size // 8  # exactly the size of each i, i+8k class inside pod

        if leaf_down + self.leaf_uplinks > P:
            raise ValueError("Radix violation: leaf_down + leaf_uplinks > 128.")

        # Pick #spines per pod = leaf_uplinks so each leaf can connect to all pod spines (full bipartite).
        spines_per_pod = self.leaf_uplinks

        # Spine southbound ports used = 8 (one to each leaf) if full bipartite,
        # but since we connect each leaf to every spine, each spine sees 8 leaf links.
        spine_south = L  # = 8
        spine_north = P - spine_south  # available if we need super-spines

        # ---- Assign IDs ----
        next_id = n_gpus

        # HBI hubs
        num_hbis = n_gpus // H
        hbi_ids = np.arange(next_id, next_id + num_hbis, dtype=np.int32)
        next_id += num_hbis

        # Leaves and spines per pod
        leaf_ids_by_pod: list[np.ndarray] = []
        spine_ids_by_pod: list[np.ndarray] = []
        for _ in range(num_pods):
            leaf_ids = np.arange(next_id, next_id + L, dtype=np.int32)
            next_id += L
            spine_ids = np.arange(next_id, next_id + spines_per_pod, dtype=np.int32)
            next_id += spines_per_pod
            leaf_ids_by_pod.append(leaf_ids)
            spine_ids_by_pod.append(spine_ids)

        # Super-spines only if >1 pod
        super_spine_ids = np.array([], dtype=np.int32)
        if num_pods > 1:
            total_north_links = num_pods * spines_per_pod * spine_north
            num_super = int(np.ceil(total_north_links / P))
            super_spine_ids = np.arange(next_id, next_id + num_super, dtype=np.int32)
            next_id += num_super

        total_nodes = int(next_id)

        # ---- Edges ----
        edges_u: list[int] = []
        edges_v: list[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) GPU <-> HBI hub (distance-2 inside HBI)
        gpu_to_hbi = np.empty(n_gpus, dtype=np.int32)
        for g in range(n_gpus):
            hid = int(hbi_ids[g // H])
            gpu_to_hbi[g] = hid
            add_bidir(int(g), hid)

        # (2) GPU <-> Leaf (stride-8 leaf rule inside each pod)
        # Leaf index in pod = (local_gpu_index % 8)
        for g in range(n_gpus):
            pod = g // pod_size
            if pod >= num_pods:
                pod = num_pods - 1  # defensive
            local = g - pod * pod_size
            leaf_idx = int(local % 8)  # 0..7
            lid = int(leaf_ids_by_pod[pod][leaf_idx])
            add_bidir(int(g), lid)

        # (3) Leaf <-> Spine within each pod (full bipartite)
        # Ensures any two leaves in the pod are at distance 2 via a spine,
        # thus GPU diameter within pod <= 4.
        for pod in range(num_pods):
            for lid in leaf_ids_by_pod[pod].tolist():
                for sid in spine_ids_by_pod[pod].tolist():
                    add_bidir(int(lid), int(sid))

        # (4) Spine <-> Super-spine (if multiple pods)
        # Round-robin assign each spine's northbound ports to super-spines.
        if num_pods > 1:
            ss_list = super_spine_ids.tolist()
            rr = 0
            for pod in range(num_pods):
                for sid in spine_ids_by_pod[pod].tolist():
                    for _ in range(spine_north):
                        ssid = ss_list[rr % len(ss_list)]
                        rr += 1
                        add_bidir(int(sid), int(ssid))

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )

        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "HBILeafSpineSuperSpine128",
            "num_gpus": n_gpus,
            "switch_ports": P,
            "hbi_size": H,
            "num_hbis": num_hbis,
            "pod_size": pod_size,
            "pods": num_pods,
            "leaves_per_pod": L,
            "spines_per_pod": spines_per_pod,
            "leaf_rule": "GPU g connects to leaf ( (g % pod_size) % 8 ) within its pod",
            "port_splits": {
                "leaf": {"downlinks_to_gpus": leaf_down, "uplinks_to_spines": self.leaf_uplinks, "total": leaf_down + self.leaf_uplinks},
                "spine": {"south_to_leaves": spine_south, "north_to_super": (spine_north if num_pods > 1 else 0), "total": spine_south + (spine_north if num_pods > 1 else 0)},
                "super_spine": {"ports": P},
                "hbi": {"to_gpus": H, "note": "modeled as hub; not port-limited"},
            },
            "counts": {
                "total_nodes_including_hbi_and_switches": total_nodes,
                "super_spines": int(super_spine_ids.size),
            },
            "ids": {
                "hbi_ids": hbi_ids,
                "leaf_ids_by_pod": leaf_ids_by_pod,
                "spine_ids_by_pod": spine_ids_by_pod,
                "super_spine_ids": super_spine_ids,
            },
        }

        res = TopologyBuildResult(edges=edges, attrs={"capacity": capacity, "weight": weight}, meta=meta)
        self._cached_build = res
        return res

    def convert_to_networkx(self) -> nx.DiGraph:
        res = self._cached_build if self._cached_build is not None else self.build_topology()

        edges = res.edges
        cap = res.attrs.get("capacity")
        w = res.attrs.get("weight")

        order = np.lexsort((edges[:, 1], edges[:, 0]))

        G = nx.DiGraph()
        total_nodes = res.meta.get("counts", {}).get("total_nodes_including_hbi_and_switches", None)
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

        G.graph["meta"] = res.meta
        return G