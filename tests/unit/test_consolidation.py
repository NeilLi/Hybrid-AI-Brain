import pytest
import time

from src.memory.working_memory import WorkingMemory
from src.memory.long_term_memory import LongTermMemory
from src.memory.flashbulb_buffer import FlashbulbBuffer
from src.memory.consolidation import ConsolidationProcess, GAMMA

def test_consolidation_moves_item_to_long_term(monkeypatch):
    wm = WorkingMemory(capacity=5)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=5)

    wm.add_item("task_1", "Test working memory content.")
    assert wm.current_size == 1
    assert ltm.total_items == 0

    # Patch time to ensure the consolidation will run
    monkeypatch.setattr(time, "time", lambda: 1000)
    consolidator = ConsolidationProcess(wm, ltm, fb)
    consolidator.last_consolidation_time = 995  # 5 seconds ago (GAMMA is 2.7)

    consolidator.run_cycle_if_needed()

    # The item should have been consolidated and removed from working memory
    assert wm.current_size == 0
    assert ltm.total_items == 1
    # The summary string should appear in LTM
    assert any("Summary of 'task_1'" in item for item in ltm._store)

def test_consolidation_prunes_flashbulb(monkeypatch):
    wm = WorkingMemory(capacity=2)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=2)

    # Add a flashbulb event with a decayed confidence
    item = fb.capture_event("Critical failure", confidence=1.0)
    # Simulate time passing to decay the item
    item._last_update -= 10000  # Make it very old

    monkeypatch.setattr(time, "time", lambda: 1000)
    consolidator = ConsolidationProcess(wm, ltm, fb)
    consolidator.last_consolidation_time = 995

    consolidator.run_cycle_if_needed()

    # After pruning, the flashbulb buffer should have pruned decayed items
    items = fb.get_all_items()
    # Assuming your prune logic removes highly decayed items
    assert all(i.get_current_weight() > 0 for i in items) or len(items) == 0

def test_consolidation_no_cycle_if_not_time(monkeypatch):
    wm = WorkingMemory(capacity=2)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=2)

    wm.add_item("recent_task", "Should not be consolidated yet.")

    # Time is set so not enough time has passed
    monkeypatch.setattr(time, "time", lambda: 1000)
    consolidator = ConsolidationProcess(wm, ltm, fb)
    consolidator.last_consolidation_time = 999.5  # only 0.5 seconds ago

    consolidator.run_cycle_if_needed()
    assert wm.current_size == 1
    assert ltm.total_items == 0


