from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


class HyperX(Topology):
    """
    Port-limited HyperX topology.

    Model:
      - Endpoints (GPUs) are nodes [0..num_nodes-1]
      - Routers are additional nodes [num_nodes .. num_nodes + R - 1]

    HyperX parameters:
      - dims: tuple of dimension sizes (S1, S2, ..., SD) such that
              R = product(Sd) routers.
      - endpoints_per_router: p
      - connect_all_in_dimension: if True, in each dimension, routers form a clique
        within each "slice" (classic HyperX complete connectivity per dimension).
        If False, you can later add k-nearest / stride variants (not implemented here).

    Port budget per router:
      p + degree_interconnect <= router_ports

    Where (for connect_all_in_dimension=True):
      degree_interconnect = sum_d (Sd - 1)

    Sizing behavior:
      - If dims is None, we auto-choose a factorization close to a cube (D=3 by default)
        that yields enough routers to host all endpoints given p.
      - If dims is provided but too small to host num_nodes endpoints, we raise.

    Notes:
      - This builds a directed graph with bidirectional physical links (2 directed edges per link).
      - Routing weights are uniform by default (weight=link_weight).
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
        auto_dims_D: int = 3,  # used only if dims is None
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
        """
        Heuristic: factor target_routers into D dimensions as balanced as possible.
        We do this by distributing prime factors to the currently smallest dimension.
        """
        dims = [1] * D
        for p in sorted(HyperX._prime_factors(target_routers), reverse=True):
            i = int(np.argmin(dims))
            dims[i] *= p
        dims.sort(reverse=True)
        return tuple(int(x) for x in dims)

    def _auto_choose_dims(self, needed_routers: int) -> Tuple[int, ...]:
        """
        Choose dims with product >= needed_routers by picking a balanced factorization
        of some R >= needed_routers. We choose the smallest R >= needed_routers that
        is easy to factor (simple increment search).
        """
        R = max(1, int(needed_routers))
        # Find a nearby R that factors nicely (small search window)
        for candidate in range(R, R + 512):
            dims = self._balanced_dims(candidate, self.auto_dims_D)
            if np.prod(dims) == candidate:
                return dims
        # Fallback: just use the exact needed_routers factorization (even if skewed)
        return self._balanced_dims(R, self.auto_dims_D)

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        P = self.router_ports
        p = self.p

        # Minimum routers required to host endpoints
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

        # Degree (router-router) for full connectivity per dimension
        degree_inter = int(sum((s - 1) for s in dims))

        # Port constraint: p endpoints + interconnect degree <= P
        if p + degree_inter > P:
            raise ValueError(
                f"Port constraint violated: endpoints_per_router + sum(Sd-1) = {p} + {degree_inter} = {p+degree_inter} > {P}. "
                "Reduce dims sizes, reduce endpoints_per_router, or increase router_ports."
            )

        # Node IDs
        gpu_ids = np.arange(0, n, dtype=np.int32)
        router_base = n
        router_ids = np.arange(router_base, router_base + R, dtype=np.int32)
        total_nodes = int(router_base + R)

        # Map GPU -> router (contiguous fill over routers)
        gpu_to_router = np.empty(n, dtype=np.int32)
        for i in range(n):
            gpu_to_router[i] = int(router_ids[(i // p) % R])

        # Coordinate mapping helpers
        # Linear index -> coordinates and back
        strides = []
        prod = 1
        for s in reversed(dims):
            strides.append(prod)
            prod *= s
        strides = list(reversed(strides))  # stride per dimension

        def idx_to_coord(idx: int) -> Tuple[int, ...]:
            coord = []
            x = idx
            for s, st in zip(dims, strides):
                c = (x // st) % s
                coord.append(int(c))
            return tuple(coord)

        def coord_to_idx(coord: Tuple[int, ...]) -> int:
            x = 0
            for c, st in zip(coord, strides):
                x += int(c) * int(st)
            return int(x)

        # Build edges (bidirectional physical links -> 2 directed edges per undirected)
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) Endpoint <-> Router
        for gpu in range(n):
            add_bidir(int(gpu), int(gpu_to_router[gpu]))

        # (2) Router-router links:
        # For each router, for each dimension d, connect to all routers that differ only in coord[d].
        # To avoid duplicating undirected links, only add if src_idx < dst_idx.
        undirected = set()
        for src_idx in range(R):
            c = list(idx_to_coord(src_idx))
            src = int(router_base + src_idx)
            for d, Sd in enumerate(dims):
                original = c[d]
                for new_val in range(Sd):
                    if new_val == original:
                        continue
                    c[d] = new_val
                    dst_idx = coord_to_idx(tuple(c))
                    dst = int(router_base + dst_idx)
                    key = (min(src, dst), max(src, dst))
                    if key in undirected:
                        continue
                    undirected.add(key)
                    add_bidir(src, dst)
                c[d] = original  # restore

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )
        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "HyperX (port-limited, complete-dimension)",
            "num_endpoints": n,
            "router_ports": P,
            "params": {
                "dims": dims,
                "routers": R,
                "endpoints_per_router": p,
                "connect_all_in_dimension": True,
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
            },
            "node_ranges": {
                "endpoints": (0, n - 1),
                "routers": (router_base, router_base + R - 1),
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