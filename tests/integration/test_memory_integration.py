import time
from src.memory.working_memory import WorkingMemory
from src.memory.long_term_memory import LongTermMemory
from src.memory.flashbulb_buffer import FlashbulbBuffer
from src.memory.consolidation import ConsolidationProcess

def test_memory_consolidation_cycle():
    wm = WorkingMemory(capacity=3)
    ltm = LongTermMemory()
    fb = FlashbulbBuffer(capacity=3)
    consolidator = ConsolidationProcess(wm, ltm, fb)

    wm.add_item("key1", "value1")
    wm.add_item("key2", "value2")
    fb.capture_event("Event1", confidence=0.9)

    time.sleep(0.1)  # Simulate time passage for triggering
    consolidator.run_cycle_if_needed()
    # After consolidation, LTM should have at least one item
    assert ltm.total_items >= 1

