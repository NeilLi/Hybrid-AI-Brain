import pytest
import time
from src.memory.flashbulb_buffer import FlashbulbBuffer, MemoryItem

def test_flashbulb_buffer_add_and_get():
    buffer = FlashbulbBuffer(capacity=3)
    item1 = buffer.capture_event("event 1", confidence=0.8)
    item2 = buffer.capture_event("event 2", confidence=0.6)
    item3 = buffer.capture_event("event 3", confidence=0.95)

    # Should be three items in buffer
    items = buffer.get_all_items()
    assert len(items) == 3
    assert items[0].event == "event 1"
    assert items[1].event == "event 2"
    assert items[2].event == "event 3"

def test_flashbulb_buffer_capacity_eviction():
    buffer = FlashbulbBuffer(capacity=2)
    buffer.capture_event("event 1", confidence=0.5)
    buffer.capture_event("event 2", confidence=0.7)
    buffer.capture_event("event 3", confidence=0.9)  # Should evict "event 1"

    items = buffer.get_all_items()
    assert len(items) == 2
    events = [item.event for item in items]
    assert "event 1" not in events
    assert "event 2" in events
    assert "event 3" in events

def test_flashbulb_buffer_prune_decayed_items():
    buffer = FlashbulbBuffer(capacity=3)
    item = buffer.capture_event("recent event", confidence=1.0)
    # Simulate a decayed item by setting _last_update in the past
    item._last_update -= 1000
    buffer.prune_decayed_items()
    items = buffer.get_all_items()
    # Depending on implementation, decayed events should be pruned.
    # Here, if your prune logic checks time or confidence decay, adjust as needed.
    # For now, we expect only non-decayed events remain.
    # This will pass if prune_decayed_items actually removes decayed ones.
    assert all(i.get_current_weight() > 0 for i in items)

def test_flashbulb_buffer_clear():
    buffer = FlashbulbBuffer(capacity=2)
    buffer.capture_event("event A", confidence=0.9)
    buffer.capture_event("event B", confidence=0.4)
    buffer.clear()
    assert len(buffer.get_all_items()) == 0

