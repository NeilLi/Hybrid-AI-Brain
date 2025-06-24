#!/usr/bin/env python3
"""
src/memory/consolidation.py

Implements the memory consolidation process, managing the flow of
information between the working, long-term, and flashbulb memory tiers.
"""

import time
import logging
from typing import Optional, Callable, Any, List

from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory
from .flashbulb_buffer import FlashbulbBuffer, MemoryItem

# --- Constants ---
GAMMA = 2.7  # Periodic consolidation trigger interval (sec) [source: 105, 113]

logger = logging.getLogger("hybrid_ai_brain.memory.consolidation")
logging.basicConfig(level=logging.INFO)

class ConsolidationProcess:
    """
    Periodically manages information flow between memory tiers (Section 7.4).
    - Consolidates from WM to LTM (filtered/summarized).
    - Prunes decayed flashbulb items.
    - Moves highly significant flashbulb items to LTM.
    """
    def __init__(
        self,
        working_memory: WorkingMemory,
        long_term_memory: LongTermMemory,
        flashbulb_buffer: FlashbulbBuffer,
        summarizer: Optional[Callable[[str, Any], str]] = None,
        wm_to_ltm_batch_size: int = 2
    ):
        self.working_memory = working_memory
        self.long_term_memory = long_term_memory
        self.flashbulb_buffer = flashbulb_buffer
        self.last_consolidation_time = time.time()
        self.summarizer = summarizer or self._default_summarizer
        self.wm_to_ltm_batch_size = wm_to_ltm_batch_size
        logger.info("ConsolidationProcess initialized.")

    def _default_summarizer(self, key: str, value: Any) -> str:
        return f"Summary of '{key}': Content starts with '{str(value)[:50]}...'"

    def run_cycle_if_needed(self):
        """
        Runs the consolidation cycle if enough time has elapsed (γ seconds).
        """
        current_time = time.time()
        if (current_time - self.last_consolidation_time) < GAMMA:
            logger.debug("Consolidation skipped: not enough time elapsed.")
            return

        logger.info(f"\nConsolidationProcess: Triggered (γ={GAMMA}s). Running cycle...")
        self.last_consolidation_time = current_time

        # 1. Consolidate (summarize & transfer) items from Working to Long-Term Memory
        wm_items = list(self.working_memory.as_dict().items())
        items_to_consolidate = wm_items[:self.wm_to_ltm_batch_size]
        for key, value in items_to_consolidate:
            summary = self.summarizer(key, value)
            self.long_term_memory.add_knowledge(summary)
            self.working_memory._store.pop(key, None)
            logger.info(f"  - Consolidated item '{key}' from WM → LTM.")

        # 2. Prune decayed items from Flashbulb Buffer
        self.flashbulb_buffer.prune_decayed_items()

        # 3. Consolidate highly weighted flashbulb items to Long-Term Memory
        for item in self.flashbulb_buffer.get_all_items():
            if item.get_current_weight() > 0.8:
                summary = self.summarizer("flashbulb", item.content)
                self.long_term_memory.add_knowledge(summary)
                logger.info("  - Consolidated highly significant flashbulb item to LTM.")

        logger.info("ConsolidationProcess: Cycle complete.")

def main():
    """Demonstrates the ConsolidationProcess."""
    logger.info("====== Memory Layer: ConsolidationProcess Demo ======")

    wm = WorkingMemory(capacity=10)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=10)

    consolidator = ConsolidationProcess(wm, ltm, fb)

    # Add items to memory as if a task is running
    logger.info("--- Simulating a running task ---")
    wm.add_item("current_query", "What is the GDP per capita of Argentina?")
    wm.add_item("intermediate_result", {"winner": "Argentina"})
    fb.capture_event("Agent 5 failed to respond", confidence=0.95)

    logger.info(f"Memory state before consolidation:\n  {wm}\n  {ltm}\n  {fb}")

    # Simulate time passing to trigger consolidation
    logger.info(f"--- Simulating {GAMMA} seconds of time passing ---")
    time.sleep(GAMMA)

    # Run the consolidation cycle
    consolidator.run_cycle_if_needed()

    logger.info(f"Memory state after consolidation:\n  {wm}\n  {ltm}\n  {fb}")
    logger.info("Note: WM→LTM transfer and flashbulb item processed.")

    logger.info("✅ consolidation.py executed successfully!")

if __name__ == "__main__":
    main()
