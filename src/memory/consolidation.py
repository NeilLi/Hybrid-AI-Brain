#!/usr/bin/env python3
"""
src/memory/consolidation.py

Implements the ConsolidationProcess that moves data between memory tiers
according to the M/G/1 queueing model with decay parameter λd = 0.45
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("hybrid_ai_brain.consolidation")

# Constants from the paper
GAMMA_INTERVAL = 1.0  # Consolidation cycle interval in seconds
MIN_WORKING_MEMORY_ITEMS = 2  # Minimum items before consolidation triggers
MIN_FLASHBULB_WEIGHT = 0.1  # Minimum weight for flashbulb items to consolidate

@dataclass
class ConsolidationMetrics:
    """Tracks consolidation performance metrics"""
    total_cycles: int = 0
    items_consolidated: int = 0
    last_cycle_time: float = 0.0
    last_cycle_duration: float = 0.0

class ConsolidationProcess:
    """
    Manages the transfer of information between memory tiers:
    - Working Memory (M) → Long-Term Memory (L)
    - Flashbulb Buffer (F) → Long-Term Memory (L)
    
    Implements the M/G/1 queueing model with λd = 0.45 decay parameter
    """
    
    def __init__(self, working_memory, long_term_memory, flashbulb_buffer):
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory  
        self.flashbulb_buffer = flashbulb_buffer
        self.last_consolidation_time = 0.0
        self.metrics = ConsolidationMetrics()
        
        logger.info("ConsolidationProcess initialized with GAMMA_INTERVAL=%.2f seconds", GAMMA_INTERVAL)
    
    def should_consolidate(self) -> bool:
        """
        Determines if consolidation should run based on:
        1. Time interval (GAMMA_INTERVAL)
        2. Working memory load
        3. Flashbulb buffer content
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_consolidation_time
        
        # Time-based trigger
        time_trigger = time_elapsed >= GAMMA_INTERVAL
        
        # Load-based triggers
        wm_trigger = self.working_memory.current_size >= MIN_WORKING_MEMORY_ITEMS
        fb_trigger = len(self.flashbulb_buffer.get_all_items()) > 0
        
        should_run = time_trigger and (wm_trigger or fb_trigger)
        
        logger.debug(
            "Consolidation check: time_elapsed=%.2f, wm_size=%d, fb_items=%d, should_run=%s",
            time_elapsed, self.working_memory.current_size, 
            len(self.flashbulb_buffer.get_all_items()), should_run
        )
        
        return should_run
    
    def consolidate_working_memory(self) -> int:
        """
        Consolidates items from Working Memory to Long-Term Memory.
        Returns number of items consolidated.
        """
        items_consolidated = 0
        wm_snapshot = self.working_memory.as_dict()
        
        logger.info("Consolidating %d items from Working Memory", len(wm_snapshot))
        
        for key, value in wm_snapshot.items():
            # Convert to string format suitable for LTM storage
            if isinstance(value, str):
                knowledge_item = f"Key: {key}, Value: {value}"
            else:
                knowledge_item = f"Key: {key}, Value: {str(value)}"
            
            # Add metadata about the consolidation
            metadata = {
                "source": "working_memory",
                "original_key": key,
                "consolidation_time": time.time(),
                "consolidation_cycle": self.metrics.total_cycles + 1
            }
            
            logger.debug("Adding to LTM: %s", knowledge_item)
            self.long_term_memory.add_knowledge(knowledge_item, metadata)
            items_consolidated += 1
        
        # Clear working memory after successful consolidation
        if items_consolidated > 0:
            self.working_memory.clear()
            logger.info("Cleared Working Memory after consolidating %d items", items_consolidated)
        
        return items_consolidated
    
    def consolidate_flashbulb_buffer(self) -> int:
        """
        Consolidates significant items from Flashbulb Buffer to Long-Term Memory.
        Only consolidates items above minimum weight threshold.
        Returns number of items consolidated.
        """
        items_consolidated = 0
        current_time = time.time()
        
        # Get items that are still significant enough to consolidate
        significant_items = [
            item for item in self.flashbulb_buffer.get_all_items()
            if item.get_current_weight(current_time) >= MIN_FLASHBULB_WEIGHT
        ]
        
        logger.info("Consolidating %d significant items from Flashbulb Buffer", len(significant_items))
        
        for item in significant_items:
            # Convert flashbulb item to knowledge string
            current_weight = item.get_current_weight(current_time)
            age = current_time - item.timestamp
            
            knowledge_item = f"Flashbulb Event: {str(item.content)} (confidence: {item.initial_confidence:.2f}, age: {age:.2f}s)"
            
            # Add metadata about the flashbulb event
            metadata = {
                "source": "flashbulb_buffer",
                "initial_confidence": item.initial_confidence,
                "timestamp": item.timestamp,
                "current_weight": current_weight,
                "consolidation_time": current_time,
                "consolidation_cycle": self.metrics.total_cycles + 1
            }
            
            logger.debug("Adding flashbulb to LTM: %s", knowledge_item)
            self.long_term_memory.add_knowledge(knowledge_item, metadata)
            items_consolidated += 1
        
        # Prune decayed items from flashbulb buffer
        initial_fb_size = len(self.flashbulb_buffer.get_all_items())
        self.flashbulb_buffer.prune_decayed_items(weight_threshold=MIN_FLASHBULB_WEIGHT)
        pruned_count = initial_fb_size - len(self.flashbulb_buffer.get_all_items())
        
        if pruned_count > 0:
            logger.info("Pruned %d decayed items from Flashbulb Buffer", pruned_count)
        
        return items_consolidated
    
    def run_cycle_if_needed(self) -> bool:
        """
        Runs a consolidation cycle if conditions are met.
        Returns True if consolidation was performed, False otherwise.
        """
        if not self.should_consolidate():
            return False
        
        cycle_start_time = time.time()
        logger.info("Starting consolidation cycle #%d", self.metrics.total_cycles + 1)
        
        try:
            # Consolidate from both memory sources
            wm_consolidated = self.consolidate_working_memory()
            fb_consolidated = self.consolidate_flashbulb_buffer()
            
            total_consolidated = wm_consolidated + fb_consolidated
            
            # Update metrics
            self.metrics.total_cycles += 1
            self.metrics.items_consolidated += total_consolidated
            self.metrics.last_cycle_time = cycle_start_time
            self.metrics.last_cycle_duration = time.time() - cycle_start_time
            self.last_consolidation_time = cycle_start_time
            
            logger.info(
                "Consolidation cycle #%d completed: %d items consolidated (WM: %d, FB: %d) in %.3f seconds",
                self.metrics.total_cycles, total_consolidated, wm_consolidated, fb_consolidated,
                self.metrics.last_cycle_duration
            )
            
            # Verify consolidation worked
            ltm_items_after = self.long_term_memory.total_items
            logger.info("Long-Term Memory now contains %d total items", ltm_items_after)
            
            return True
            
        except Exception as e:
            logger.error("Consolidation cycle failed: %s", str(e))
            return False
    
    def force_consolidation(self) -> bool:
        """
        Forces a consolidation cycle regardless of timing conditions.
        Useful for testing and manual triggers.
        """
        logger.info("Forcing consolidation cycle (bypassing timing conditions)")
        
        # Temporarily reset timing to force consolidation
        original_time = self.last_consolidation_time
        self.last_consolidation_time = 0.0
        
        result = self.run_cycle_if_needed()
        
        # Restore original timing if consolidation failed
        if not result:
            self.last_consolidation_time = original_time
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Returns current consolidation status and metrics"""
        current_time = time.time()
        time_since_last = current_time - self.last_consolidation_time
        
        return {
            "total_cycles": self.metrics.total_cycles,
            "items_consolidated": self.metrics.items_consolidated,
            "time_since_last_cycle": time_since_last,
            "last_cycle_duration": self.metrics.last_cycle_duration,
            "should_consolidate_now": self.should_consolidate(),
            "working_memory_size": self.working_memory.current_size,
            "flashbulb_buffer_items": len(self.flashbulb_buffer.get_all_items()),
            "long_term_memory_items": self.long_term_memory.total_items
        }
    
    def __repr__(self) -> str:
        return (f"ConsolidationProcess(cycles={self.metrics.total_cycles}, "
                f"consolidated={self.metrics.items_consolidated}, "
                f"ltm_items={self.long_term_memory.total_items})")


# --- Demo/Test Block ---
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import memory components
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.memory.working_memory import WorkingMemory
    from src.memory.long_term_memory import LongTermMemory  
    from src.memory.flashbulb_buffer import FlashbulbBuffer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("====== ConsolidationProcess Demo ======")
    
    # Initialize memory components
    wm = WorkingMemory(capacity=5)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=5)
    consolidator = ConsolidationProcess(wm, ltm, fb)
    
    print(f"Initial state: {consolidator}")
    print(f"Status: {consolidator.get_status()}")
    
    # Add some data to working memory
    print("\n--- Adding data to Working Memory ---")
    wm.add_item("task1", "Process user request #123")
    wm.add_item("task2", "Update system configuration")
    wm.add_item("result1", "Request processed successfully")
    
    # Add some events to flashbulb buffer
    print("\n--- Adding events to Flashbulb Buffer ---")
    fb.capture_event("Critical error detected in module X", confidence=0.9)
    fb.capture_event("New optimization discovered", confidence=0.8)
    
    print(f"Before consolidation: WM={wm.current_size} items, FB={len(fb.get_all_items())} items, LTM={ltm.total_items} items")
    
    # Wait and run consolidation
    print("\n--- Running consolidation ---")
    time.sleep(1.1)  # Wait for GAMMA_INTERVAL
    
    success = consolidator.run_cycle_if_needed()
    print(f"Consolidation success: {success}")
    print(f"After consolidation: WM={wm.current_size} items, FB={len(fb.get_all_items())} items, LTM={ltm.total_items} items")
    
    # Test force consolidation
    print("\n--- Testing force consolidation ---")
    wm.add_item("emergency", "System restart required")
    fb.capture_event("User reported bug in feature Y", confidence=0.7)
    
    success = consolidator.force_consolidation()
    print(f"Force consolidation success: {success}")
    print(f"Final state: WM={wm.current_size} items, FB={len(fb.get_all_items())} items, LTM={ltm.total_items} items")
    
    # Show final status
    print(f"\nFinal consolidator state: {consolidator}")
    print(f"Final status: {consolidator.get_status()}")
    
    # Test retrieval from LTM
    print("\n--- Testing LTM retrieval ---")
    results = ltm.retrieve_relevant_knowledge("task", top_k=3)
    print(f"Retrieved {len(results)} items for query 'task':")
    for i, item in enumerate(results, 1):
        print(f"  {i}. {item}")
    
    print("\n====== Demo completed successfully! ======")