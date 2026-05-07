from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult
try:
    # Package-style import (when graph_overloading is imported as a package)
    from .. import hyperx_dimension_structures as hds
except ImportError:
    # Script-style fallback (when running from graph_overloading directory)
    import hyperx_dimension_structures as hds



class HyperX(Topology):
    """
    Megatron-aware HyperX:

    - One router per (tp_leaf, pp, dp) coordinate
    - 8 GPUs per router (tp dimension chunk)
    - router-router edges implement HyperX "clique per dimension"
    """

    def __init__(
        self,
        num_nodes: int,  # GPUs
        *,
        router_ports: int = 64,
        endpoints_per_router: int = 8,
        transport_csv_path: Optional[str] = None,  # <-- new
        aggregate_tp_groups: bool = False,
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if router_ports <= 1:
            raise ValueError("router_ports must be > 1")
        if endpoints_per_router != 8:
            raise ValueError("This HyperX model is fixed to 8 GPUs per leaf/router.")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.router_ports = int(router_ports)
        self.p = int(endpoints_per_router)  # 8
        self.transport_csv_path = transport_csv_path
        self.aggregate_tp_groups = bool(aggregate_tp_groups)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        P = self.router_ports
        p = self.p

        # --- derive dims from Megatron CSV filename when provided ---
        if self.transport_csv_path is None:
            raise ValueError(
                "transport_csv_path must be provided for Megatron-aware HyperX dims "
                "(or extend this to accept manual dims)."
            )

        # Use hds helpers if you already have them for parsing. Fallback to your regex parser if not.
        # Expected return: world_size, tp, pp
        if hasattr(hds, "parse_megatron_csv_filename"):
            parsed = hds.parse_megatron_csv_filename(self.transport_csv_path)
            if parsed is None:
                raise ValueError(f"Could not parse world_size/tp/pp from: {self.transport_csv_path}")
            world_size, tp, pp = parsed
        else:
            # minimal fallback parse (adjust to your naming conventions if needed)
            import re
            m = re.search(r"-world_size(\d+)-tp(\d+)-pp(\d+)", self.transport_csv_path)
            if not m:
                raise ValueError(f"Could not parse world_size/tp/pp from: {self.transport_csv_path}")
            world_size, tp, pp = int(m.group(1)), int(m.group(2)), int(m.group(3))

        if world_size != n:
            # You can choose to allow mismatch and override n <- world_size instead.
            raise ValueError(f"num_nodes={n} does not match parsed world_size={world_size}")

        denom = tp * pp
        if denom <= 0 or (world_size % denom) != 0:
            raise ValueError(f"world_size={world_size} not divisible by tp*pp={denom}")
        dp = world_size // denom

        if tp % p != 0:
            raise ValueError(f"tp={tp} must be divisible by endpoints_per_router={p} (8)")

        # dims = (tp_leaf_count, pp, dp)
        # When aggregating TP groups, collapse TP axis to size 1 so consecutive
        # HBI groups (8-GPU chunks) under the same TP group share one switch.
        # This keeps intra-HBI distance=1 via direct GPU mesh, while different
        # HBI groups under the same switch are distance=2 (gpu -> switch -> gpu).
        Sx = 1 if self.aggregate_tp_groups else (tp // p)
        Sy, Sz = pp, dp
        R = int(Sx * Sy * Sz)

        # Port check for clique-per-dimension HyperX
        degree_inter = (Sx - 1) + (Sy - 1) + (Sz - 1)
        endpoints_per_router_effective = tp if self.aggregate_tp_groups else p
        if endpoints_per_router_effective + degree_inter > P:
            raise ValueError(
                f"Port constraint violated: endpoints_per_router={endpoints_per_router_effective} "
                f"+ interconnect_degree={degree_inter} "
                f"=> {endpoints_per_router_effective + degree_inter} > router_ports={P}."
            )

        router_base = n
        total_nodes = router_base + R

        # Coordinate indexing (z fastest)
        def coord_to_idx(coord: Tuple[int, int, int]) -> int:
            x, y, z = coord
            return int((x * Sy + y) * Sz + z)

        def idx_to_coord(ridx: int) -> Tuple[int, int, int]:
            x = ridx // (Sy * Sz)
            rem = ridx % (Sy * Sz)
            y = rem // Sz
            z = rem % Sz
            return int(x), int(y), int(z)

        # --- GPU rank -> router mapping using Megatron coords ---
        rank_coords = hds.megatron_rank_to_hyperx_coords(
            tp=tp,
            pp=pp,
            dp=dp,
            gpus_per_leaf=p,
            aggregate_tp_groups=self.aggregate_tp_groups,
        )

        gpu_to_router = np.empty(n, dtype=np.int32)
        router_id_to_coord: Dict[int, Tuple[int, int, int]] = {}
        router_idx_to_coord: Dict[int, Tuple[int, int, int]] = {}

        for ridx in range(R):
            c = idx_to_coord(ridx)
            router_idx_to_coord[ridx] = c
            router_id_to_coord[router_base + ridx] = c

        # assign each rank to its router based on (x,y,z)
        for gpu in range(n):
            x, y, z = rank_coords[gpu]
            ridx = coord_to_idx((x, y, z))
            gpu_to_router[gpu] = int(router_base + ridx)

        # group GPUs per router for intra-leaf full mesh
        router_to_gpus: Dict[int, List[int]] = {}
        for gpu in range(n):
            r = int(gpu_to_router[gpu])
            router_to_gpus.setdefault(r, []).append(gpu)

        # --- build edges ---
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) GPU <-> Router edges
        for gpu in range(n):
            add_bidir(int(gpu), int(gpu_to_router[gpu]))

        # (1b) Intra-router GPU full mesh (should be size 8 each, but tolerate partial)
        for r, gpus in router_to_gpus.items():
            gpus_sorted = sorted(gpus)
            for i in range(len(gpus_sorted)):
                for j in range(i + 1, len(gpus_sorted)):
                    add_bidir(int(gpus_sorted[i]), int(gpus_sorted[j]))

        # (2) Router-router HyperX links (clique per dimension)
        undirected = set()
        for src_idx in range(R):
            x, y, z = idx_to_coord(src_idx)
            src = router_base + src_idx

            # vary x
            for nx_ in range(Sx):
                if nx_ == x:
                    continue
                dst = router_base + coord_to_idx((nx_, y, z))
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

            # vary y
            for ny_ in range(Sy):
                if ny_ == y:
                    continue
                dst = router_base + coord_to_idx((x, ny_, z))
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

            # vary z
            for nz_ in range(Sz):
                if nz_ == z:
                    continue
                dst = router_base + coord_to_idx((x, y, nz_))
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

        edges = np.column_stack((np.asarray(edges_u, np.int32), np.asarray(edges_v, np.int32)))
        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "HyperX-3D Megatron-driven (tp_leaf, pp, dp)",
            "transport_csv_path": self.transport_csv_path,
            "megatron": {
                "world_size": world_size,
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "gpus_per_leaf": p,
                "aggregate_tp_groups": self.aggregate_tp_groups,
            },
            "params": {"dims": (Sx, Sy, Sz), "routers": R, "router_ports": P},
            "coord_order": "idx=(x*Sy + y)*Sz + z  (z fastest)",
            "coord_maps": {
                "router_idx_to_coord": router_idx_to_coord,
                "router_id_to_coord": router_id_to_coord,
            },
            "gpu_to_router": gpu_to_router,
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
        total_nodes = res.meta.get("megatron", {}).get("world_size", self.num_nodes) + res.meta["params"]["routers"]
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