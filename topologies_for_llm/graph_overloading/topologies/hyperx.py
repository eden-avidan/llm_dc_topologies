from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


class HyperX(Topology):
    """
    HyperX topology (classic complete connectivity per dimension), with direct GPU-GPU links
    within each router/HBI group.

    Structure:
      - endpoints_per_router GPUs are connected to each router
      - GPUs within the same router have direct links (full mesh)
      - Routers are connected in a HyperX pattern (complete graph per dimension)

    Distance model:
      - Same router (consecutive endpoints_per_router GPUs): 1 hop (direct GPU-GPU link)
      - Different routers: (number of differing dimensions) + 2 hops
        - e.g., differ in 1 dim: GPU → Router1 → Router2 → GPU = 3 hops
        - e.g., differ in 2 dims: GPU → R1 → R2 → R3 → GPU = 4 hops
        - e.g., differ in 3 dims: GPU → R1 → R2 → R3 → R4 → GPU = 5 hops

    Model:
      - Endpoints (GPUs): nodes [0..num_nodes-1]
      - Routers (HBIs): nodes [num_nodes .. num_nodes + R - 1]
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        router_ports: int = 64,
        endpoints_per_router: int = 8,
        dims: Tuple[int, ...] | None = None,
        connect_all_in_dimension: bool = True,
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
        auto_dims_D: int = 3,
    ) -> None:
        super().__init__(num_nodes)

        if router_ports <= 1:
            raise ValueError("router_ports must be > 1")
        if endpoints_per_router <= 0:
            raise ValueError("endpoints_per_router must be > 0")
        if dims is not None and (len(dims) == 0 or any(int(s) <= 0 for s in dims)):
            raise ValueError("dims must be a tuple of positive ints")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")
        if auto_dims_D < 1:
            raise ValueError("auto_dims_D must be >= 1")

        self.router_ports = int(router_ports)
        self.p = int(endpoints_per_router)
        self.dims = tuple(int(x) for x in dims) if dims is not None else None
        self.connect_all_in_dimension = bool(connect_all_in_dimension)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)
        self.auto_dims_D = int(auto_dims_D)

        self._cached_build: TopologyBuildResult | None = None

    @staticmethod
    def _prime_factors(n: int) -> List[int]:
        f: List[int] = []
        x = n
        d = 2
        while d * d <= x:
            while x % d == 0:
                f.append(d)
                x //= d
            d += 1
        if x > 1:
            f.append(x)
        return f

    @staticmethod
    def _balanced_dims(target_routers: int, D: int) -> Tuple[int, ...]:
        dims = [1] * D
        for p in sorted(HyperX._prime_factors(target_routers), reverse=True):
            i = int(np.argmin(dims))
            dims[i] *= p
        dims.sort(reverse=True)
        return tuple(int(x) for x in dims)

    def _auto_choose_dims(self, needed_routers: int) -> Tuple[int, ...]:
        """
        Autoselect dims (shape) given needed router count.

        Uses balanced factorization to distribute routers across D dimensions
        as evenly as possible.
        """
        R = int(max(1, needed_routers))
        return self._balanced_dims(R, self.auto_dims_D)

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        P = self.router_ports
        p = self.p

        routers_min = int(np.ceil(n / p))

        dims = self.dims if self.dims is not None else self._auto_choose_dims(routers_min)
        R = int(np.prod(dims))

        if R < routers_min:
            raise ValueError(
                f"dims product={R} routers is insufficient for num_nodes={n} with p={p} "
                f"(need at least {routers_min} routers)."
            )

        if not self.connect_all_in_dimension:
            raise NotImplementedError("Only connect_all_in_dimension=True is implemented.")

        # Degree for classic HyperX with K=1
        degree_inter = int(sum((s - 1) for s in dims))

        if p + degree_inter > P:
            raise ValueError(
                f"Port constraint violated: {p} + sum(Sd-1)={degree_inter} => {p + degree_inter} > {P}. "
                "Reduce dims, reduce endpoints_per_router, or increase router_ports."
            )

        router_base = n
        router_ids = np.arange(router_base, router_base + R, dtype=np.int32)
        total_nodes = int(router_base + R)

        # GPU -> router mapping: contiguous blocks of size p (NO modulo wrap)
        # This matches your heatmap assumption: each HBI connects exactly 8 GPUs.
        gpu_to_router = np.empty(n, dtype=np.int32)
        for i in range(n):
            ridx = i // p
            if ridx >= R:
                raise ValueError(
                    f"Not enough routers for contiguous mapping: GPU {i} maps to router index {ridx}, "
                    f"but only {R} routers exist. Increase dims product or reduce endpoints_per_router."
                )
            gpu_to_router[i] = int(router_ids[ridx])

        # Coordinate helpers for routers 0..R-1 (row-major indexing for shape=dims)
        strides: List[int] = []
        prod = 1
        for s in reversed(dims):
            strides.append(prod)
            prod *= s
        strides = list(reversed(strides))

        def idx_to_coord(idx: int) -> Tuple[int, ...]:
            return tuple(int((idx // st) % s) for s, st in zip(dims, strides))

        def coord_to_idx(coord: Tuple[int, ...]) -> int:
            return int(sum(int(c) * int(st) for c, st in zip(coord, strides)))

        # --- explicit coordinate maps ---
        router_idx_to_coord: Dict[int, Tuple[int, ...]] = {}
        router_id_to_coord: Dict[int, Tuple[int, ...]] = {}
        for ridx in range(R):
            c = idx_to_coord(ridx)
            router_idx_to_coord[ridx] = c
            router_id_to_coord[int(router_base + ridx)] = c

        gpu_id_to_router_idx = np.empty(n, dtype=np.int32)
        gpu_id_to_coord: Dict[int, Tuple[int, ...]] = {}
        for gpu in range(n):
            ridx = gpu // p  # contiguous blocks
            gpu_id_to_router_idx[gpu] = ridx
            rc = router_idx_to_coord[ridx]
            i = gpu % p
            gpu_id_to_coord[gpu] = (*rc, int(i))  # (x,y,z,i)


        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) GPU <-> Router
        for gpu in range(n):
            add_bidir(int(gpu), int(gpu_to_router[gpu]))

        # (1b) Direct GPU <-> GPU links within each router group (full mesh)
        # This creates direct distance-1 links between consecutive endpoints_per_router GPUs
        for ridx in range(R):
            base_gpu = ridx * p
            # Create full mesh within the router group
            for i in range(p):
                for j in range(i + 1, p):
                    gpu_i = base_gpu + i
                    gpu_j = base_gpu + j
                    if gpu_i < n and gpu_j < n:
                        add_bidir(gpu_i, gpu_j)

        # (2) Router-router HyperX links (differ in exactly 1 coordinate)
        undirected = set()
        for src_idx in range(R):
            c = list(idx_to_coord(src_idx))
            src = int(router_base + src_idx)

            for d, Sd in enumerate(dims):
                orig = c[d]
                for new_val in range(Sd):
                    if new_val == orig:
                        continue
                    c[d] = new_val
                    dst_idx = coord_to_idx(tuple(c))
                    dst = int(router_base + dst_idx)

                    key = (min(src, dst), max(src, dst))
                    if key in undirected:
                        continue
                    undirected.add(key)
                    add_bidir(src, dst)

                c[d] = orig

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )
        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "HyperX (complete-dimension, with direct GPU-GPU links)",
            "num_endpoints": n,
            "router_ports": P,
            "params": {
                "dims": dims,
                "routers": R,
                "endpoints_per_router": p,
                "connect_all_in_dimension": True,
            },
            "distance_model": {
                "same_router": 1,
                "differ_1_dim": 3,  # 1 + 2
                "differ_2_dims": 4,  # 2 + 2
                "differ_3_dims": 5,  # 3 + 2
                "formula": "differing_dimensions + 2",
            },
            "port_accounting_per_router": {
                "endpoint_ports": p,
                "interconnect_ports": degree_inter,
                "total_used": p + degree_inter,
                "budget": P,
            },
            "counts": {
                "total_nodes_including_routers": total_nodes,
                "routers": R,
                "endpoints": n,
                "direct_gpu_links_per_router": p * (p - 1) // 2,  # full mesh = C(p,2)
            },
            "node_ranges": {
                "endpoints": (0, n - 1),
                "routers": (router_base, router_base + R - 1),
            },
            "gpu_to_router": gpu_to_router,
        }

        meta["coord_maps"] = {
            "router_idx_to_coord": router_idx_to_coord,
            "router_id_to_coord": router_id_to_coord,
            "gpu_id_to_coord": gpu_id_to_coord,
            "gpu_id_to_router_idx": gpu_id_to_router_idx,
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