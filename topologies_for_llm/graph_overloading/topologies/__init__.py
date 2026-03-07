"""
Topology package for graph-overloading experiments.
Contains the abstract base topology class and concrete topology builders
(Fat Tree, HyperX, Dragonfly+) used by the main analyzer.
"""

from .abstract_topology import Topology, TopologyBuildResult

__all__ = ['Topology', 'TopologyBuildResult']
