#!/usr/bin/env python3
"""
src/memory/working_memory.py

Defines the Working Memory (M) component of the hierarchical memory system.
"""

from collections import OrderedDict
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger("hybrid_ai_brain.working_memory")
logging.basicConfig(level=logging.INFO)

class WorkingMemory:
    """
    Working Memory (Short-Term Memory): Fast-access, LRU, capacity-limited buffer.
    """
    def __init__(self, capacity: int = 200, on_evict: Optional[Callable[[str, Any], None]] = None):
        if capacity <= 0:
            raise ValueError("Working Memory capacity must be positive.")
        self.capacity: int = capacity
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.on_evict = on_evict  # Optional eviction callback
        logger.info(f"WorkingMemory initialized with capacity φ={self.capacity}.")

    def add_item(self, key: str, value: Any) -> None:
        """
        Adds or updates an item. Evicts LRU if over capacity.
        """
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            evicted_key, evicted_value = self._store.popitem(last=False)
            logger.info(f"WorkingMemory: Capacity reached. Evicted LRU item '{evicted_key}'.")
            if self.on_evict:
                self.on_evict(evicted_key, evicted_value)
        logger.debug(f"WorkingMemory: Added/updated item '{key}'.")

    def add_items(self, items: Dict[str, Any]) -> None:
        """
        Bulk add items (maintains LRU order, may cause multiple evictions).
        """
        for k, v in items.items():
            self.add_item(k, v)

    def get_item(self, key: str) -> Optional[Any]:
        """
        Retrieves and marks item as recently used.
        """
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def get_items(self, keys: List[str]) -> Dict[str, Any]:
        """
        Bulk get multiple items.
        """
        return {k: self.get_item(k) for k in keys if k in self._store}

    def clear(self) -> None:
        self._store.clear()
        logger.info("WorkingMemory: Cleared all items.")

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a dict snapshot of current memory state (ordered, LRU to MRU).
        """
        return dict(self._store)

    def resize(self, new_capacity: int) -> None:
        """
        Adjust capacity. Truncates if downsizing.
        """
        if new_capacity <= 0:
            raise ValueError("New capacity must be positive.")
        while len(self._store) > new_capacity:
            evicted_key, evicted_value = self._store.popitem(last=False)
            logger.info(f"WorkingMemory: Downsizing. Evicted LRU item '{evicted_key}'.")
            if self.on_evict:
                self.on_evict(evicted_key, evicted_value)
        self.capacity = new_capacity
        logger.info(f"WorkingMemory: Resized to capacity {self.capacity}.")

    @property
    def current_size(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={self.current_size}, capacity={self.capacity})"

# --- Demo Block ---
if __name__ == "__main__":
    wm = WorkingMemory(capacity=3)
    wm.add_item("a", 1)
    wm.add_item("b", 2)
    wm.add_item("c", 3)
    print(wm)
    wm.add_item("d", 4)  # Should evict 'a'
    print(wm)
    wm.get_item("b")     # Mark 'b' as MRU
    wm.add_item("e", 5)  # Should evict 'c'
    print(wm)
    print("Items:", wm.as_dict())
    wm.resize(2)
    print("After resize:", wm)
    wm.clear()
    print("Cleared:", wm)
