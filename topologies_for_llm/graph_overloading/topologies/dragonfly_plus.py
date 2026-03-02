from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult

class DragonflyPlus(Topology):
    """
    A practical, port-limited Dragonfly+ style topology.

    Model:
      - Endpoints (GPUs) are nodes [0..num_nodes-1]
      - Routers are additional nodes [num_nodes .. num_nodes + g*a - 1]

    Parameters:
      P: router_ports (same router type everywhere)
      a: routers_per_group
      p: endpoints_per_router
      h: global_links_per_router
      Local connectivity: within each group, routers form a clique (a-1 local links/router).
      Global connectivity: deterministic "DF+"-like mapping that spreads links across groups
                          to increase diversity. (One of many valid Dragonfly+ wirings.)

    Port constraint per router:
      p + (a-1) + h <= P

    Sizing:
      - If a is not provided, we pick the largest a that fits the port budget
        given p and h (this minimizes group count for a given num_nodes).
      - g = ceil(num_nodes / (a*p))  (minimum groups to host endpoints)
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        router_ports: int = 64,
        endpoints_per_router: int = 8,     # p
        global_links_per_router: int = 8,  # h
        routers_per_group: int | None = None,  # a (optional; auto-sized if None)
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if router_ports <= 1:
            raise ValueError("router_ports must be > 1")
        if endpoints_per_router <= 0:
            raise ValueError("endpoints_per_router must be > 0")
        if global_links_per_router < 0:
            raise ValueError("global_links_per_router must be >= 0")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.router_ports = int(router_ports)
        self.p = int(endpoints_per_router)
        self.h = int(global_links_per_router)
        self.a = int(routers_per_group) if routers_per_group is not None else None

        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        self._cached_build: TopologyBuildResult | None = None

    def _choose_a(self) -> int:
        """Choose largest a that fits port budget, given p and h."""
        # Need at least 1 router per group
        max_a = self.router_ports - self.p - self.h + 1  # because (a-1) + p + h <= P
        if max_a < 1:
            raise ValueError(
                f"Port budget impossible: need p+h <= P. Got p={self.p}, h={self.h}, P={self.router_ports}"
            )
        return max_a

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        P = self.router_ports
        p = self.p
        h = self.h
        a = self.a if self.a is not None else self._choose_a()

        # Validate port constraint
        if p + (a - 1) + h > P:
            raise ValueError(
                f"Port constraint violated: p+(a-1)+h={p+(a-1)+h} > P={P}. "
                "Reduce routers_per_group (a), endpoints_per_router (p), or global_links_per_router (h)."
            )

        # Minimum groups to host endpoints
        routers_total_min = int(np.ceil(n / p))
        g = int(np.ceil(routers_total_min / a))
        routers_total = g * a

        # If g==1, global links can't go anywhere meaningful; we’ll just omit them.
        effective_h = h if g >= 2 else 0

        # Node IDs
        gpu_ids = np.arange(0, n, dtype=np.int32)
        router_base = n
        router_ids = np.arange(router_base, router_base + routers_total, dtype=np.int32)
        total_nodes = int(router_base + routers_total)

        # Router indexing helpers
        # group_id in [0..g-1], router_in_group in [0..a-1]
        def rid(group_id: int, r_in_group: int) -> int:
            return int(router_base + group_id * a + r_in_group)

        # Map GPUs -> routers (contiguous fill)
        gpu_to_router = np.empty(n, dtype=np.int32)
        for i in range(n):
            router_idx = i // p
            gpu_to_router[i] = int(router_ids[router_idx])

        # Build edges (bidirectional physical links -> 2 directed edges)
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) Endpoint <-> Router
        for gpu in range(n):
            add_bidir(int(gpu), int(gpu_to_router[gpu]))

        # (2) Local links within each group: clique among routers in the group
        for group_id in range(g):
            routers = [rid(group_id, r) for r in range(a)]
            for i in range(a):
                for j in range(i + 1, a):
                    add_bidir(routers[i], routers[j])

        # (3) Global links: deterministic DF+-like spread across groups.
        # We create exactly effective_h global links per router.
        # To avoid duplicates, track undirected (min,max) pairs.
        global_pairs = set()

        if effective_h > 0:
            for group_id in range(g):
                for r in range(a):
                    src = rid(group_id, r)
                    for t in range(effective_h):
                        # Choose a destination group that depends on (group,router,port)
                        # offset >=1 ensures we don't pick same group unless g==1 (handled above)
                        dest_group = (group_id + 1 + r * effective_h + t) % g
                        dest_router = (r + t) % a
                        dst = rid(dest_group, dest_router)

                        if src == dst:
                            continue
                        key = (min(src, dst), max(src, dst))
                        if key in global_pairs:
                            continue
                        global_pairs.add(key)
                        add_bidir(src, dst)

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )

        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "Dragonfly+ (port-limited, clique-local, deterministic-global)",
            "num_endpoints": n,
            "router_ports": P,
            "params": {
                "routers_per_group": a,
                "endpoints_per_router": p,
                "global_links_per_router": h,
                "effective_global_links_per_router": effective_h,
                "groups": g,
                "routers_total": routers_total,
            },
            "port_accounting_per_router": {
                "endpoint_ports": p,
                "local_ports": a - 1,
                "global_ports": effective_h,
                "total_used": p + (a - 1) + effective_h,
                "budget": P,
            },
            "counts": {
                "total_nodes_including_routers": total_nodes,
                "routers": routers_total,
                "endpoints": n,
            },
            "node_ranges": {
                "endpoints": (0, n - 1),
                "routers": (router_base, router_base + routers_total - 1),
            },
            "gpu_to_router": gpu_to_router,
        }

        res = TopologyBuildResult(edges=edges, attrs={"capacity": capacity, "weight": weight}, meta=meta)
        self._cached_build = res
        return res

    def convert_to_networkx(self) -> nx.DiGraph:
        """
        Convert built topology to NetworkX DiGraph with deterministic adjacency order.
        """
        res = self._cached_build if self._cached_build is not None else self.build_topology()
        edges = res.edges
        cap = res.attrs.get("capacity")
        w = res.attrs.get("weight")

        # deterministic order by (u,v)
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