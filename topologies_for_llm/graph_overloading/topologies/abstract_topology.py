from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import networkx as nx


@dataclass(frozen=True)
class TopologyBuildResult:
    """
    Minimal, backend-agnostic representation of a topology.

    edges: array of shape (m, 2) where each row is (u, v)
    attrs: optional edge attributes stored as numpy arrays of shape (m,)
           e.g. attrs["capacity"], attrs["weight"]
    meta:  optional metadata (mappings, parameters, etc.)
    """
    edges: np.ndarray
    attrs: Dict[str, np.ndarray]
    meta: Dict[str, Any]


class Topology(ABC):
    """
    Abstract base class for network topologies.

    - Requires `num_nodes` at construction time.
    - Stores it as `self.num_nodes`.
    - Forces subclasses to implement `build_topology()`.
    """

    def __init__(self, num_nodes: int) -> None:
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise ValueError(f"num_nodes must be a positive int, got {num_nodes!r}")
        self.num_nodes: int = num_nodes

    @property
    def n(self) -> int:
        """Convenience alias for number of nodes."""
        return self.num_nodes

    @abstractmethod
    def build_topology(self) -> TopologyBuildResult:
        """
        Construct and return the topology.

        Subclasses must return a TopologyBuildResult containing at least:
        - edges: np.ndarray (m,2) int32/int64
        and may include:
        - attrs: e.g. 'capacity', 'weight'
        - meta: arbitrary helpful metadata
        """
        raise NotImplementedError

    @abstractmethod
    def convert_to_networkx(self) -> nx.DiGraph:
        raise NotImplementedError