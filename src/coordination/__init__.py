#!/usr/bin/env python3
"""
src/coordination/__init__.py

Exposes the primary classes of the coordination layer, including the GNN,
the bio-optimizers, and the conflict resolution mechanism.
"""

from .gnn_coordinator import GNNCoordinator
from .bio_optimizer import BioOptimizer
from .conflict_resolver import ConflictResolver

__all__ = [
    "GNNCoordinator",
    "BioOptimizer",
    "ConflictResolver",
]
