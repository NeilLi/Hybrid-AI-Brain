#!/usr/bin/env python3
"""
tests/unit/test_flashbulb_buffer.py

Fixed tests for the FlashbulbBuffer implementation that match the actual API.
"""

import pytest
import time
import numpy as np
from src.memory.flashbulb_buffer import FlashbulbBuffer, MemoryItem, LAMBDA_D

def test_flashbulb_buffer_add_and_get():
    """Test adding events and retrieving them from the buffer."""
    buffer = FlashbulbBuffer(capacity=3)
    
    # capture_event doesn't return anything, so we call it without assignment
    buffer.capture_event("event 1", confidence=0.8)
    buffer.capture_event("event 2", confidence=0.6)
    buffer.capture_event("event 3", confidence=0.95)

    # Should be three items in buffer
    items = buffer.get_all_items()
    assert len(items) == 3
    
    # Check the content attribute (not data or event)
    contents = [item.content for item in items]
    assert "event 1" in contents
    assert "event 2" in contents
    assert "event 3" in contents
    
    # Check confidence values are stored correctly
    confidences = [item.initial_confidence for item in items]
    assert 0.8 in confidences
    assert 0.6 in confidences
    assert 0.95 in confidences

def test_flashbulb_buffer_capacity_eviction():
    """Test that the buffer evicts lowest-weight items when at capacity."""
    buffer = FlashbulbBuffer(capacity=2)
    
    buffer.capture_event("event 1", confidence=0.5)
    buffer.capture_event("event 2", confidence=0.7)
    
    # Allow some time to pass so the first item has lower weight
    time.sleep(0.1)
    
    buffer.capture_event("event 3", confidence=0.9)  # Should evict "event 1" (lowest confidence)

    items = buffer.get_all_items()
    assert len(items) == 2
    
    contents = [item.content for item in items]
    # The lowest confidence item should have been evicted
    assert "event 1" not in contents
    assert "event 2" in contents
    assert "event 3" in contents

def test_flashbulb_buffer_memory_item_weight_decay():
    """Test that MemoryItem weights decay over time according to the exponential model."""
    # Create a memory item directly
    item = MemoryItem(content="test event", initial_confidence=1.0)
    
    # Check initial weight (should be close to confidence)
    initial_weight = item.get_current_weight()
    assert abs(initial_weight - 1.0) < 0.01
    
    # Simulate time passing by providing a future timestamp
    future_time = item.timestamp + 1.0  # 1 second later
    decayed_weight = item.get_current_weight(future_time)
    
    # Weight should have decayed according to: w = c * e^(-λ_d * t)
    expected_weight = 1.0 * np.exp(-LAMBDA_D * 1.0)
    assert abs(decayed_weight - expected_weight) < 0.01
    
    # Weight should be less than initial
    assert decayed_weight < initial_weight

def test_flashbulb_buffer_prune_decayed_items():
    """Test pruning of items that have decayed below threshold."""
    buffer = FlashbulbBuffer(capacity=3)
    
    # Add an event
    buffer.capture_event("recent event", confidence=1.0)
    
    # Manually create an old item by modifying its timestamp
    old_item = MemoryItem(content="old event", initial_confidence=1.0)
    old_item.timestamp = time.time() - 10.0  # 10 seconds ago
    buffer._buffer.append(old_item)
    
    # Add another recent event
    buffer.capture_event("another recent event", confidence=0.9)
    
    assert len(buffer.get_all_items()) == 3
    
    # Prune with a threshold that should remove the old item
    # After 10 seconds with λ_d = 0.45, weight = 1.0 * e^(-0.45 * 10) ≈ 0.011
    buffer.prune_decayed_items(weight_threshold=0.02)
    
    items = buffer.get_all_items()
    # The old item should be pruned, leaving 2 items
    assert len(items) <= 2
    
    # All remaining items should have weight above threshold
    assert all(item.get_current_weight() > 0.02 for item in items)

def test_flashbulb_buffer_clear():
    """Test clearing all items from the buffer."""
    buffer = FlashbulbBuffer(capacity=2)
    buffer.capture_event("event A", confidence=0.9)
    buffer.capture_event("event B", confidence=0.4)
    
    assert len(buffer.get_all_items()) == 2
    
    buffer.clear()
    assert len(buffer.get_all_items()) == 0

def test_flashbulb_buffer_weight_sum():
    """Test the current_weight_sum property."""
    buffer = FlashbulbBuffer(capacity=3)
    
    # Empty buffer should have zero weight
    assert buffer.current_weight_sum == 0.0
    
    # Add events and check weight sum
    buffer.capture_event("event 1", confidence=0.8)
    buffer.capture_event("event 2", confidence=0.6)
    
    weight_sum = buffer.current_weight_sum
    assert weight_sum > 0
    
    # Weight sum should be approximately the sum of individual weights
    items = buffer.get_all_items()
    manual_sum = sum(item.get_current_weight() for item in items)
    assert abs(weight_sum - manual_sum) < 0.001

def test_flashbulb_buffer_ignore_invalid_confidence():
    """Test that events with non-positive confidence are ignored."""
    buffer = FlashbulbBuffer(capacity=3)
    
    # These should be ignored
    buffer.capture_event("invalid event 1", confidence=0.0)
    buffer.capture_event("invalid event 2", confidence=-0.5)
    
    # This should be added
    buffer.capture_event("valid event", confidence=0.5)
    
    items = buffer.get_all_items()
    assert len(items) == 1
    assert items[0].content == "valid event"

def test_flashbulb_buffer_eviction_order():
    """Test that eviction prioritizes items with lowest current weight."""
    buffer = FlashbulbBuffer(capacity=2)
    
    # Add first item with high confidence
    buffer.capture_event("high confidence event", confidence=0.9)
    
    # Wait a bit to let it decay slightly
    time.sleep(0.1)
    
    # Add second item with lower confidence
    buffer.capture_event("low confidence event", confidence=0.3)
    
    # Add third item - should evict the one with lowest current weight
    buffer.capture_event("new high confidence event", confidence=0.95)
    
    items = buffer.get_all_items()
    assert len(items) == 2
    
    # The low confidence event should have been evicted
    contents = [item.content for item in items]
    assert "low confidence event" not in contents
    assert "high confidence event" in contents
    assert "new high confidence event" in contents

def test_memory_item_string_representation():
    """Test the string representation of MemoryItem."""
    item = MemoryItem(content="test event", initial_confidence=0.75)
    
    repr_str = repr(item)
    assert "MemoryItem" in repr_str
    assert "confidence=0.75" in repr_str
    assert "age=" in repr_str
    assert "current_weight=" in repr_str

def test_flashbulb_buffer_string_representation():
    """Test the string representation of FlashbulbBuffer."""
    buffer = FlashbulbBuffer(capacity=5)
    buffer.capture_event("test event", confidence=0.8)
    
    repr_str = repr(buffer)
    assert "FlashbulbBuffer" in repr_str
    assert "items=1" in repr_str
    assert "capacity=5" in repr_str
    assert "total_weight=" in repr_str

if __name__ == "__main__":
    print("Running FlashbulbBuffer tests...")
    
    # Run all tests
    test_functions = [
        test_flashbulb_buffer_add_and_get,
        test_flashbulb_buffer_capacity_eviction,
        test_flashbulb_buffer_memory_item_weight_decay,
        test_flashbulb_buffer_prune_decayed_items,
        test_flashbulb_buffer_clear,
        test_flashbulb_buffer_weight_sum,
        test_flashbulb_buffer_ignore_invalid_confidence,
        test_flashbulb_buffer_eviction_order,
        test_memory_item_string_representation,
        test_flashbulb_buffer_string_representation
    ]
    
    passed = 0
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
    
    print(f"\n{passed}/{len(test_functions)} tests passed!")
    
    if passed == len(test_functions):
        print("🎉 All FlashbulbBuffer tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")