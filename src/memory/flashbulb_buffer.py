#!/usr/bin/env python3
"""
src/memory/flashbulb_buffer.py

Defines the Flashbulb Buffer (F), a specialized memory component for
capturing significant events with high fidelity and decaying importance.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional

# --- Constants from the paper's theoretical framework ---
# These are critical for the memory freshness guarantees.
LAMBDA_D = 0.45  # Optimized memory decay rate (λ_d) [source: 112, 116, 834]
W_MAX = 50       # Maximum cumulative weight before flush is considered (W_max) [source: 102]

@dataclass
class MemoryItem:
    """
    Represents a single item in the flashbulb buffer, such as a critical
    error log or a breakthrough discovery. Its importance decays over time.
    """
    content: Any
    initial_confidence: float
    timestamp: float = field(default_factory=time.time)

    def get_current_weight(self, current_time: Optional[float] = None) -> float:
        """
        Calculates the item's current weight based on exponential decay.
        This directly implements Equation 3 from the paper: w_i(t) = c_i * e^(-λ_d*t)
        [source: 108]

        Args:
            current_time: The current time. If None, time.time() is used.

        Returns:
            The calculated weight of the memory item.
        """
        current_time = current_time or time.time()
        age = current_time - self.timestamp
        return self.initial_confidence * np.exp(-LAMBDA_D * age)

    def __repr__(self) -> str:
        return (f"MemoryItem(confidence={self.initial_confidence:.2f}, "
                f"age={time.time() - self.timestamp:.2f}s, "
                f"current_weight={self.get_current_weight():.2f})")

class FlashbulbBuffer:
    """
    Manages a collection of salient MemoryItems.
    This buffer ensures that important, rare events are not lost and can be
    recalled in full detail for analysis or learning.
    [source: 696]
    """
    def __init__(self, capacity: int = 100):
        """
        Initializes the Flashbulb Buffer.

        Args:
            capacity (int): The maximum number of items the buffer can hold (θ).
                            [source: 102]
        """
        self.capacity = capacity
        self._buffer: List[MemoryItem] = []
        print(f"FlashbulbBuffer initialized with capacity θ={self.capacity}.")

    def capture_event(self, content: Any, confidence: float):
        """
        Captures a new salient event and adds it to the buffer.
        If the buffer is at capacity, it makes space by removing the item
        with the lowest current weight.
        [source: 698]
        """
        if confidence <= 0:
            print("FlashbulbBuffer: Ignoring event with non-positive confidence.")
            return

        if len(self._buffer) >= self.capacity:
            # Evict the item with the lowest current weight to make space
            self._buffer.sort(key=lambda item: item.get_current_weight(), reverse=True)
            evicted_item = self._buffer.pop()
            print(f"FlashbulbBuffer: Capacity reached. Evicted lowest-weight item: {evicted_item}")

        item = MemoryItem(content=content, initial_confidence=confidence)
        self._buffer.append(item)
        print(f"FlashbulbBuffer: Captured new event with confidence {confidence:.2f}.")

    def get_all_items(self) -> List[MemoryItem]:
        """Returns all items currently in the buffer."""
        return self._buffer

    def prune_decayed_items(self, weight_threshold: float = 0.01):
        """
        Removes items from the buffer whose weight has decayed below a threshold.
        This is typically called by the consolidation process.
        """
        initial_count = len(self._buffer)
        current_time = time.time()
        self._buffer = [
            item for item in self._buffer
            if item.get_current_weight(current_time) > weight_threshold
        ]
        pruned_count = initial_count - len(self._buffer)
        if pruned_count > 0:
            print(f"FlashbulbBuffer: Pruned {pruned_count} decayed item(s).")
    
    @property
    def current_weight_sum(self) -> float:
        """Calculates the sum of all current weights in the buffer."""
        return sum(item.get_current_weight() for item in self._buffer)
        
    def __repr__(self) -> str:
        return (f"FlashbulbBuffer(items={len(self._buffer)}, "
                f"capacity={self.capacity}, "
                f"total_weight={self.current_weight_sum:.2f}/{W_MAX})")
    def clear(self):
        self._buffer.clear()

def main():
    """Demonstrates the functionality of the FlashbulbBuffer and MemoryItem."""
    print("====== Memory Layer: FlashbulbBuffer Demo ======")
    buffer = FlashbulbBuffer(capacity=5)

    # 1. Capture some events
    print("\n--- Capturing Events ---")
    buffer.capture_event("Critical system error #1234", confidence=0.9)
    buffer.capture_event("Novel solution found for task XYZ", confidence=0.8)
    time.sleep(1) # Let some time pass for decay
    buffer.capture_event("Unexpected user query pattern detected", confidence=0.6)
    
    print("\nCurrent Buffer State:")
    for item in buffer.get_all_items():
        print(f"  - {item}")
    print(buffer)
    
    # 2. Simulate time passing to see weights decay
    print("\n--- Simulating 2 seconds of time passing ---")
    time.sleep(2)
    
    print("\nBuffer State After Decay:")
    for item in buffer.get_all_items():
        print(f"  - {item}")
    print(buffer)

    # 3. Prune items that have decayed significantly
    buffer.prune_decayed_items(weight_threshold=0.2)
    print("\nBuffer State After Pruning (threshold=0.2):")
    for item in buffer.get_all_items():
        print(f"  - {item}")

    print("\n====================================================")
    print("✅ flashbulb_buffer.py executed successfully!")

if __name__ == "__main__":
    main()
