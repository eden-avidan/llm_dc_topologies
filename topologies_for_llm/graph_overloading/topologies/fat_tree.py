from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import networkx as nx

from .abstract_topology import Topology, TopologyBuildResult


class FatTree(Topology):
    """
    Implements the exact behavior spec:

    Hierarchy:
      GPU (within node via HBI mesh) -> Rail (leaf) -> Spine -> SuperSpine

    Grouping:
      - 8 GPUs per compute node: node_id = gpu_id // 8, inner index j = gpu_id % 8
      - 64 nodes per "block"/rail-domain: block_id = node_id // 64

    Links and required shortest-path hop costs (with all edge weights = 1):
      - Same node: direct HBI GPU<->GPU edges (clique) => cost 1
      - Same block AND same j: GPU -> rail(block,j) -> GPU => cost 2
      - Same block AND different j: GPU -> rail -> spine(block) -> rail -> GPU => cost 4
      - Different blocks: GPU -> rail -> spine(blockA) -> superspine -> spine(blockB) -> rail -> GPU => cost 6

    Port constraints (enforced by construction):
      - Rail: downlinks = number of nodes in its block (<=64), uplinks = 1 (<=64)
      - Spine: downlinks = 8 rails, uplinks = 1
      - SuperSpine: downlinks = #blocks; must be <= 128, else we raise.
    """

    def __init__(
        self,
        num_nodes: int,  # number of GPUs
        *,
        switch_ports: int = 128,
        node_size: int = 8,
        nodes_per_block: int = 64,
        link_capacity: float = 1.0,
        link_weight: float = 1.0,
    ) -> None:
        super().__init__(num_nodes)

        if switch_ports != 128:
            raise ValueError("This topology is specifically for radix-128 switches.")
        if node_size != 8:
            raise ValueError("This topology is specifically for 8-GPU compute nodes.")
        if nodes_per_block != 64:
            raise ValueError("This topology is specifically for 64-node blocks (rail-domains).")

        if self.num_nodes % node_size != 0:
            raise ValueError("num_nodes must be divisible by 8 (node size).")

        if link_capacity <= 0:
            raise ValueError("link_capacity must be > 0")
        if link_weight <= 0:
            raise ValueError("link_weight must be > 0")

        self.switch_ports = int(switch_ports)
        self.node_size = int(node_size)
        self.nodes_per_block = int(nodes_per_block)
        self.link_capacity = float(link_capacity)
        self.link_weight = float(link_weight)

        self._cached_build: TopologyBuildResult | None = None

    def build_topology(self) -> TopologyBuildResult:
        n_gpus = self.num_nodes
        H = self.node_size          # 8 GPUs per node
        B = self.nodes_per_block    # 64 nodes per block
        P = self.switch_ports       # 128

        num_nodes = n_gpus // H  # compute nodes
        num_blocks = int(np.ceil(num_nodes / B))

        # SuperSpine port check: it must connect to one spine per block.
        # (SuperSpine is a single switch in this model.)
        if num_blocks > P:
            raise ValueError(
                f"Need {num_blocks} spine links into the single super-spine, "
                f"but switch_ports=128. Reduce num_nodes or change the model."
            )

        # ---- ID allocation ----
        next_id = n_gpus

        # Rails: 8 rails per block (one per inner index j)
        num_rails = num_blocks * H
        rail_ids = np.arange(next_id, next_id + num_rails, dtype=np.int32)
        next_id += num_rails

        # One spine per block
        spine_ids = np.arange(next_id, next_id + num_blocks, dtype=np.int32)
        next_id += num_blocks

        # One superspine
        superspine_id = int(next_id)
        next_id += 1

        total_nodes = int(next_id)

        # ---- Helpers for mapping ----
        def block_id_of_gpu(g: int) -> int:
            node_id = g // H
            return node_id // B

        def inner_j_of_gpu(g: int) -> int:
            return g % H

        def rail_id_of(block_id: int, j: int) -> int:
            """
            IMPORTANT: we make rail IDs unique per block by using:
              rail_index = block_id * 8 + j
            This preserves your intended behavior (rails are block-local)
            and guarantees cross-block traffic must go via super-spine.
            """
            rail_index = block_id * H + j
            return int(rail_ids[rail_index])

        def spine_id_of(block_id: int) -> int:
            return int(spine_ids[block_id])

        # ---- Edge lists (directed graph stored as bidirectional edges) ----
        edges_u: List[int] = []
        edges_v: List[int] = []

        def add_bidir(u: int, v: int) -> None:
            edges_u.append(u); edges_v.append(v)
            edges_u.append(v); edges_v.append(u)

        # (1) Intra-node HBI: full mesh among each group of 8 GPUs => cost 1 inside node
        for node_id in range(num_nodes):
            base = node_id * H
            # GPUs in this node: [base .. base+7]
            for a in range(H):
                ga = base + a
                for b in range(a + 1, H):
                    gb = base + b
                    add_bidir(int(ga), int(gb))

        # (2) GPU <-> Rail links (downlinks)
        # GPU g connects to rail(block_id(g), j(g)) => enables 2-hop same-(block,j)
        for g in range(n_gpus):
            blk = block_id_of_gpu(g)
            j = inner_j_of_gpu(g)
            rid = rail_id_of(blk, j)
            add_bidir(int(g), rid)

        # (3) Rail <-> Spine links (uplinks inside block)
        # One spine per block connects to all 8 rails in that block => enables 4-hop same block, diff j
        for blk in range(num_blocks):
            sid = spine_id_of(blk)
            for j in range(H):
                rid = rail_id_of(blk, j)
                add_bidir(rid, sid)

        # (4) Spine <-> SuperSpine links (cross-block)
        # Connect every block spine to the single superspine => enables 6-hop cross-block
        for blk in range(num_blocks):
            sid = spine_id_of(blk)
            add_bidir(sid, superspine_id)

        # ---- Pack edges + attributes ----
        edges = np.column_stack(
            (np.asarray(edges_u, dtype=np.int32), np.asarray(edges_v, dtype=np.int32))
        )
        m = edges.shape[0]
        capacity = np.full((m,), self.link_capacity, dtype=np.float32)
        weight = np.full((m,), self.link_weight, dtype=np.float32)

        # ---- Metadata ----
        meta: Dict[str, Any] = {
            "topology": "NodeRailSpineSuperSpine128",
            "num_gpus": n_gpus,
            "switch_ports": P,
            "node_size": H,
            "nodes_per_block": B,
            "num_compute_nodes": num_nodes,
            "num_blocks": num_blocks,
            "counts": {
                "total_nodes_including_switches": total_nodes,
                "rails": int(num_rails),
                "spines": int(num_blocks),
                "superspines": 1,
            },
            "mapping_rules": {
                "node_id": "node_id = gpu_id // 8",
                "inner_index": "j = gpu_id % 8",
                "block_id": "block_id = node_id // 64",
                "rail_index": "rail_index = block_id * 8 + j",
                "gpu_to_rail": "gpu -> rail(block_id(gpu), j(gpu))",
                "rails_to_spine": "all 8 rails in block -> spine(block)",
                "spines_to_superspine": "all spines -> superspine",
            },
            "distance_goals_hops": {
                "same_node": 1,
                "same_block_same_j": 2,
                "same_block_diff_j": 4,
                "diff_block": 6,
            },
            "ids": {
                "rail_ids": rail_ids,
                "spine_ids": spine_ids,
                "superspine_id": superspine_id,
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

        G.graph["meta"] = res.meta
        return G