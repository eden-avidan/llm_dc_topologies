from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult
from .. import hyperx_dimension_structures as hds


class HyperX(Topology):
    """
    HyperX topology matching your numeric model:

    - Routers correspond 1:1 with HBI groups (8 GPUs per router/HBI)
    - GPUs within the same router have direct links (full mesh) => distance 1
    - Routers are connected so that router-to-router shortest hop count equals
      (# of differing coordinates) in 3D, i.e. a "clique per dimension" HyperX:
        * If x differs: there is a direct link to the router with same (y,z) and new x
        * Similarly for y, z
      Therefore:
        GPU_i -> RouterA -> (k router hops) -> RouterB -> GPU_j
      gives total GPU-to-GPU = k + 2 where k is number of differing dims.

    Node IDs:
      - GPUs:   0 .. n-1
      - Routers: n .. n+R-1   (R = Sx*Sy*Sz)
    """

    def __init__(
        self,
        num_nodes: int,  # GPUs
        *,
        router_ports: int = 64,
        endpoints_per_router: int = 8,  # must be 8 to match numeric model
        dims: Tuple[int, int, int] | None = None,  # optional override
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if router_ports <= 1:
            raise ValueError("router_ports must be > 1")
        if endpoints_per_router != 8:
            raise ValueError("This HyperX model is fixed to 8 GPUs per HBI/router (endpoints_per_router=8).")
        if dims is not None:
            if len(dims) != 3 or any(int(s) <= 0 for s in dims):
                raise ValueError("dims must be a 3-tuple of positive ints (Sx,Sy,Sz).")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.router_ports = int(router_ports)
        self.p = int(endpoints_per_router)  # 8
        self.dims = tuple(int(x) for x in dims) if dims is not None else None
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        P = self.router_ports
        p = self.p  # 8

        routers_min = int(ceil(n / p))

        # ---- DIM SELECTION (matches your numeric implementation) ----
        dims = self.dims if self.dims is not None else (int(ceil(n / p)), int(ceil(n / p)), int(ceil(n / p)))
        Sx, Sy, Sz = (int(dims[0]), int(dims[1]), int(dims[2]))
        R = int(Sx * Sy * Sz)

        if R < routers_min:
            raise ValueError(
                f"dims product={R} routers is insufficient for num_nodes={n} with p={p} "
                f"(need at least {routers_min} routers)."
            )

        # Classic HyperX degree with full connectivity per dimension:
        # degree_inter = (Sx-1) + (Sy-1) + (Sz-1)
        degree_inter = (Sx - 1) + (Sy - 1) + (Sz - 1)
        if p + degree_inter > P:
            raise ValueError(
                f"Port constraint violated: endpoints_per_router={p} + interconnect_degree={degree_inter} "
                f"=> {p + degree_inter} > router_ports={P}. "
                "Reduce dims (override), reduce routers (smaller n), or increase router_ports."
            )

        router_base = n
        router_ids = np.arange(router_base, router_base + R, dtype=np.int32)
        total_nodes = int(router_base + R)

        # ---- GPU -> router mapping: contiguous groups of 8 (same as your code) ----
        gpu_to_router = np.empty(n, dtype=np.int32)
        for gpu in range(n):
            ridx = gpu // p
            if ridx >= R:
                raise ValueError(
                    f"Not enough routers for contiguous mapping: GPU {gpu} maps to router index {ridx}, "
                    f"but only {R} routers exist. Increase dims product or reduce num_nodes."
                )
            gpu_to_router[gpu] = int(router_ids[ridx])

        # ---- Router coordinate maps (MUST match your numeric: z fastest row-major) ----
        # numeric: idx = (x*Sy + y)*Sz + z
        def idx_to_coord(ridx: int) -> Tuple[int, int, int]:
            x = ridx // (Sy * Sz)
            rem = ridx % (Sy * Sz)
            y = rem // Sz
            z = rem % Sz
            return int(x), int(y), int(z)

        def coord_to_idx(coord: Tuple[int, int, int]) -> int:
            x, y, z = coord
            return int((x * Sy + y) * Sz + z)

        router_idx_to_coord: Dict[int, Tuple[int, int, int]] = {}
        router_id_to_coord: Dict[int, Tuple[int, int, int]] = {}
        for ridx in range(R):
            c = idx_to_coord(ridx)
            router_idx_to_coord[ridx] = c
            router_id_to_coord[int(router_base + ridx)] = c

        gpu_id_to_router_idx = np.empty(n, dtype=np.int32)
        gpu_id_to_coord: Dict[int, Tuple[int, int, int, int]] = {}
        for gpu in range(n):
            ridx = int(gpu // p)
            gpu_id_to_router_idx[gpu] = ridx
            x, y, z = router_idx_to_coord[ridx]
            i = int(gpu % p)
            gpu_id_to_coord[gpu] = (x, y, z, i)  # include gpu-in-hbi for convenience

        # ---- Edges ----
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) GPU <-> Router edges
        for gpu in range(n):
            add_bidir(int(gpu), int(gpu_to_router[gpu]))

        # (1b) Direct GPU <-> GPU links within each router (full mesh) => distance 1 within HBI
        for ridx in range(R):
            base_gpu = ridx * p
            for i in range(p):
                for j in range(i + 1, p):
                    gi = base_gpu + i
                    gj = base_gpu + j
                    if gi < n and gj < n:
                        add_bidir(int(gi), int(gj))

        # (2) Router-router HyperX links:
        # For each router, connect to all routers that differ in exactly 1 coord
        # while keeping other coords identical. This yields router distance = #diff_dims.
        undirected = set()
        for src_idx in range(R):
            x, y, z = idx_to_coord(src_idx)
            src = int(router_base + src_idx)

            # vary x
            for nx_ in range(Sx):
                if nx_ == x:
                    continue
                dst_idx = coord_to_idx((nx_, y, z))
                dst = int(router_base + dst_idx)
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

            # vary y
            for ny_ in range(Sy):
                if ny_ == y:
                    continue
                dst_idx = coord_to_idx((x, ny_, z))
                dst = int(router_base + dst_idx)
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

            # vary z
            for nz_ in range(Sz):
                if nz_ == z:
                    continue
                dst_idx = coord_to_idx((x, y, nz_))
                dst = int(router_base + dst_idx)
                key = (min(src, dst), max(src, dst))
                if key not in undirected:
                    undirected.add(key)
                    add_bidir(src, dst)

        # ---- Pack edges + attributes ----
        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )
        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "HyperX-3D (numeric-dims, z-fastest, direct intra-HBI GPU links)",
            "num_endpoints": n,
            "router_ports": P,
            "params": {
                "dims": (Sx, Sy, Sz),
                "routers": R,
                "endpoints_per_router": p,
                "connect_all_in_dimension": True,
                "coord_order": "row-major with z fastest: idx=(x*Sy + y)*Sz + z",
                "dim_choice": "minimal_balanced_3d_dims(num_gpus, gpus_per_hbi=8)",
            },
            "distance_model": {
                "same_gpu": 0,
                "same_router": 1,
                "different_routers": "diff_dims(x,y,z) + 2",
                "examples": {
                    "diff_1_dim": 3,
                    "diff_2_dims": 4,
                    "diff_3_dims": 5,
                },
            },
            "port_accounting_per_router": {
                "endpoint_ports": p,
                "interconnect_ports": int(degree_inter),
                "total_used": int(p + degree_inter),
                "budget": P,
            },
            "counts": {
                "total_nodes_including_routers": total_nodes,
                "routers": R,
                "endpoints": n,
                "direct_gpu_links_per_router": p * (p - 1) // 2,
            },
            "node_ranges": {
                "endpoints": (0, n - 1),
                "routers": (router_base, router_base + R - 1),
            },
            "gpu_to_router": gpu_to_router,
            "coord_maps": {
                "router_idx_to_coord": router_idx_to_coord,
                "router_id_to_coord": router_id_to_coord,
                "gpu_id_to_coord": gpu_id_to_coord,
                "gpu_id_to_router_idx": gpu_id_to_router_idx,
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
        total_nodes = res.meta.get("counts", {}).get("total_nodes_including_routers")
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