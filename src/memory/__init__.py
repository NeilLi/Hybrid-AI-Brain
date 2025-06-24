#!/usr/bin/env python3
"""
src/memory/__init__.py

Exposes the primary classes of the hierarchical memory system, making it
a cohesive and importable package.
"""

from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .flashbulb_buffer import FlashbulbBuffer, MemoryItem
from .consolidation import ConsolidationProcess

__all__ = [
    "WorkingMemory",
    "LongTermMemory",
    "FlashbulbBuffer",
    "MemoryItem",
    "ConsolidationProcess",
]
