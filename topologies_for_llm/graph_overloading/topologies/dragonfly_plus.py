from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


class DragonflyPlus(Topology):
    """
    Dragonfly+ topology - paper-faithful implementation.

    Based on: "Dragonfly+: Low Cost Topology for Scaling Datacenters"
    by Shpiner et al., IEEE 2017.

    Model:
      - Endpoints (GPUs) are nodes [0..num_nodes-1]
      - Routers are additional nodes [num_nodes .. num_nodes + g*(l+s) - 1]
        - Leaf routers: Connect to p endpoints and s spine routers
        - Spine routers: Connect to l leaf routers and h inter-group global links

    Key constraints (from paper):
      - Rule-of-thumb for full bisection bandwidth: p = l = s = h (Equation 1)
      - Port balance: p = h = k/2 (Equation 2)
      - Group size: N_group = p*l = k²/4
      - Global links: ONLY spine-to-spine (never on leaves)
      - Every pair of groups has at least one direct global link

    Parameters:
      router_ports (k): Router radix
      inter_group_variant: "largest" | "medium" | "small"
        - "largest": Minimal global links (1 per group pair) - Figure 1a
        - "medium": Every spine to every group (1 link) - Figure 1b
        - "small": Every spine to every group (multiple parallel links) - Figure 1c
      parallel_links_per_spine: For "small" variant, number of parallel links

    Derived parameters (auto-calculated from k):
      p = k/2  (endpoints per leaf router)
      l = k/2  (leaf routers per group)
      s = k/2  (spine routers per group)
      h = k/2  (global links per spine router)
      g = (s*h + 1)  (number of groups for "largest" variant)
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        router_ports: int = 64,
        inter_group_variant: str = "medium",
        parallel_links_per_spine: int = 1,
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if router_ports <= 1 or router_ports % 2 != 0:
            raise ValueError("router_ports must be even and > 1")
        if inter_group_variant not in ("largest", "medium", "small"):
            raise ValueError("inter_group_variant must be 'largest', 'medium', or 'small'")
        if parallel_links_per_spine < 1:
            raise ValueError("parallel_links_per_spine must be >= 1")
        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.router_ports = int(router_ports)
        self.inter_group_variant = str(inter_group_variant)
        self.parallel_links_per_spine = int(parallel_links_per_spine)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        # Paper-specified balancing: p = l = s = h = k/2
        self.p = self.router_ports // 2
        self.l = self.router_ports // 2
        self.s = self.router_ports // 2
        self.h = self.router_ports // 2

        self._cached_build: TopologyBuildResult | None = None

    def _compute_num_groups(self, variant: str) -> int:
        """
        Compute number of groups based on variant and endpoints.

        For "largest": g = s*h + 1 (maximal network size)
        For "medium"/"small": Minimize g to fit num_nodes
        """
        s, h, p, l = self.s, self.h, self.p, self.l

        if variant == "largest":
            # Maximum groups: each spine has h global links to different groups
            return s * h + 1
        else:
            # Minimum groups to host all endpoints
            endpoints_per_group = p * l  # k²/4
            return int(np.ceil(self.num_nodes / endpoints_per_group))

    def _validate_group_count(self, g: int) -> None:
        """Validate that we can create required inter-group topology."""
        s, h = self.s, self.h

        if self.inter_group_variant == "largest":
            # Each pair of groups needs one link
            # Total global links available: g * s * h
            # Total group pairs: g * (g-1)
            # Need: g * (g-1) <= g * s * h
            #   => (g-1) <= s*h
            #   => g <= s*h + 1
            if g > s * h + 1:
                raise ValueError(
                    f"Too many groups ({g}) for 'largest' variant. "
                    f"Maximum is s*h+1 = {s*h+1} for router_ports={self.router_ports}"
                )
        elif self.inter_group_variant == "medium":
            # Every spine connects to every other group (1 link each)
            # Each spine needs (g-1) global links
            # Must have: (g-1) <= h
            if g - 1 > h:
                raise ValueError(
                    f"Too many groups ({g}) for 'medium' variant. "
                    f"Maximum is h+1 = {h+1} for router_ports={self.router_ports}"
                )
        # "small" variant can support any g <= h+1 with multiple parallel links

    def build_topology(self) -> TopologyBuildResult:
        n = self.num_nodes
        k = self.router_ports
        p, l, s, h = self.p, self.l, self.s, self.h

        # Determine number of groups
        g = self._compute_num_groups(self.inter_group_variant)
        self._validate_group_count(g)

        # Total routers
        routers_per_group = l + s
        routers_total = g * routers_per_group

        # Validate we can host all endpoints
        total_endpoints_capacity = g * l * p
        if total_endpoints_capacity < n:
            raise ValueError(
                f"Insufficient capacity: can only host {total_endpoints_capacity} endpoints, "
                f"need {n}. Increase router_ports or use different variant."
            )

        # Node IDs
        router_base = n
        total_nodes = router_base + routers_total

        # Helper functions for router IDs
        def leaf_id(group_id: int, leaf_in_group: int) -> int:
            return router_base + group_id * (l + s) + leaf_in_group

        def spine_id(group_id: int, spine_in_group: int) -> int:
            return router_base + group_id * (l + s) + l + spine_in_group

        # Map GPUs -> leaf routers (contiguous fill)
        gpu_to_router = np.empty(n, dtype=np.int32)
        for i in range(n):
            leaf_idx = i // p  # Global leaf index
            group_id = leaf_idx // l
            leaf_in_group = leaf_idx % l
            gpu_to_router[i] = leaf_id(group_id, leaf_in_group)

        # Build edges
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) Endpoint <-> Leaf Router
        for gpu in range(n):
            add_bidir(gpu, int(gpu_to_router[gpu]))

        # (2) Local bipartite links (Leaf <-> Spine within each group)
        for group_id in range(g):
            for leaf_in_group in range(l):
                for spine_in_group in range(s):
                    add_bidir(
                        leaf_id(group_id, leaf_in_group),
                        spine_id(group_id, spine_in_group)
                    )

        # (3) Global inter-group links (Spine <-> Spine across groups)
        # Implementation based on inter_group_variant

        if g == 1:
            # Single group: no global links needed
            pass

        elif self.inter_group_variant == "largest":
            # Figure 1a: Each pair of groups connected by single global link
            # Distribute links evenly across spine routers
            for group_i in range(g):
                for group_j in range(group_i + 1, g):
                    # Pick spine routers for this group pair
                    # Use deterministic mapping to distribute load
                    spine_i = (group_i + group_j) % s
                    spine_j = (group_i + group_j) % s

                    add_bidir(
                        spine_id(group_i, spine_i),
                        spine_id(group_j, spine_j)
                    )

        elif self.inter_group_variant == "medium":
            # Figure 1b: Every spine connects to every other group (1 link)
            for group_id in range(g):
                for spine_in_group in range(s):
                    for dest_group in range(g):
                        if dest_group == group_id:
                            continue

                        # Connect to corresponding spine in destination group
                        # Use offset to distribute connections
                        dest_spine = (spine_in_group + dest_group) % s

                        src = spine_id(group_id, spine_in_group)
                        dst = spine_id(dest_group, dest_spine)

                        # Only add edge once (avoid duplicates from both directions)
                        if group_id < dest_group or (group_id == dest_group and spine_in_group < dest_spine):
                            add_bidir(src, dst)

        elif self.inter_group_variant == "small":
            # Figure 1c: Every spine connects to every other group (multiple parallel links)
            for group_id in range(g):
                for spine_in_group in range(s):
                    for dest_group in range(g):
                        if dest_group == group_id:
                            continue

                        # Create multiple parallel links
                        for link_idx in range(self.parallel_links_per_spine):
                            dest_spine = (spine_in_group + dest_group + link_idx) % s

                            src = spine_id(group_id, spine_in_group)
                            dst = spine_id(dest_group, dest_spine)

                            # Add bidirectional parallel link
                            # Note: We add all parallel links (not deduping)
                            if group_id < dest_group or (group_id == dest_group and spine_in_group < dest_spine):
                                add_bidir(src, dst)

        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )

        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        meta: Dict[str, Any] = {
            "topology": "Dragonfly+ (paper-faithful, bipartite leaf-spine)",
            "paper": "Shpiner et al., IEEE 2017",
            "num_endpoints": n,
            "router_ports": k,
            "params": {
                "groups": g,
                "routers_per_group": l + s,
                "total_routers": routers_total,
                "leaves_per_group": l,
                "spines_per_group": s,
                "endpoints_per_router": p,
                "global_links_per_spine": h,
                "inter_group_variant": self.inter_group_variant,
                "parallel_links_per_spine": self.parallel_links_per_spine if self.inter_group_variant == "small" else 1,
                "balancing_rule": f"p=l=s=h={p}",
            },
            "port_accounting": {
                "leaf_router": {
                    "endpoint_ports": p,
                    "spine_ports": s,
                    "total_used": p + s,
                    "budget": k,
                },
                "spine_router": {
                    "leaf_ports": l,
                    "global_ports": h,
                    "total_used": l + h,
                    "budget": k,
                },
            },
            "counts": {
                "total_nodes_including_routers": total_nodes,
                "total_routers": routers_total,
                "leaf_routers": g * l,
                "spine_routers": g * s,
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

    def convert_to_networkx(self) -> nx.MultiDiGraph:
        """
        Convert to NetworkX MultiDiGraph to support parallel links.

        Uses MultiDiGraph (not DiGraph) because Dragonfly+ explicitly
        supports parallel links between spine routers (Figure 1c).
        """
        res = self._cached_build if self._cached_build is not None else self.build_topology()
        edges = res.edges
        cap = res.attrs.get("capacity")
        w = res.attrs.get("weight")

        # Use MultiDiGraph to preserve parallel links
        G = nx.MultiDiGraph()

        total_nodes = res.meta.get("counts", {}).get("total_nodes_including_routers")
        if isinstance(total_nodes, int):
            G.add_nodes_from(range(total_nodes))

        # Add all edges (including parallel ones)
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
