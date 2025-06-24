import pytest
from src.memory.working_memory import WorkingMemory

def test_add_and_get_item():
    wm = WorkingMemory(capacity=3)
    wm.add_item("k1", "v1")
    wm.add_item("k2", 42)
    assert wm.get_item("k1") == "v1"
    assert wm.get_item("k2") == 42

def test_eviction_policy_lru():
    wm = WorkingMemory(capacity=2)
    wm.add_item("a", 1)
    wm.add_item("b", 2)
    wm.add_item("c", 3)  # should evict "a"
    assert wm.get_item("a") is None
    assert wm.get_item("b") == 2
    assert wm.get_item("c") == 3

def test_clear_memory():
    wm = WorkingMemory(capacity=2)
    wm.add_item("x", 9)
    wm.clear()
    assert wm.current_size == 0

def test_update_marks_as_recent():
    wm = WorkingMemory(capacity=2)
    wm.add_item("a", 1)
    wm.add_item("b", 2)
    # Access "a", then add "c". "b" should now be evicted
    _ = wm.get_item("a")
    wm.add_item("c", 3)
    assert wm.get_item("b") is None
    assert wm.get_item("a") == 1
    assert wm.get_item("c") == 3

