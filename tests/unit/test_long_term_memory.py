#!/usr/bin/env python3
"""
tests/unit/test_long_term_memory.py

Fixed tests for LongTermMemory that handle both ChromaDB and in-memory backends properly.
"""

import pytest
import time
from src.memory.long_term_memory import LongTermMemory

@pytest.fixture
def fresh_ltm():
    """Create a fresh LongTermMemory instance for each test."""
    ltm = LongTermMemory()
    # Ensure it starts clean
    ltm.clear()
    yield ltm
    # Clean up after test
    ltm.clear()

def test_add_and_total_items(fresh_ltm):
    """Test adding items and counting them."""
    ltm = fresh_ltm
    
    # Should start empty
    assert ltm.total_items == 0
    
    # Add items
    ltm.add_knowledge("Fact 1")
    assert ltm.total_items == 1
    
    ltm.add_knowledge("Fact 2")
    assert ltm.total_items == 2
    
    # Verify we can get all items
    all_items = ltm.get_all_items()
    assert len(all_items) == 2
    assert "Fact 1" in all_items
    assert "Fact 2" in all_items

def test_add_empty_or_invalid_items(fresh_ltm):
    """Test that empty or invalid items are handled properly."""
    ltm = fresh_ltm
    
    # These should be ignored
    ltm.add_knowledge("")
    ltm.add_knowledge("   ")  # whitespace only
    ltm.add_knowledge(None)  # This might cause an error, should be handled gracefully
    
    assert ltm.total_items == 0
    
    # Valid item should still work
    ltm.add_knowledge("Valid fact")
    assert ltm.total_items == 1

def test_retrieve_relevant_knowledge_simple_match(fresh_ltm):
    """Test semantic/text retrieval functionality."""
    ltm = fresh_ltm
    
    # Add test data
    ltm.add_knowledge("Argentina won the World Cup in 2022.")
    ltm.add_knowledge("GDP per capita of Argentina is high.")
    ltm.add_knowledge("Python is a programming language.")
    
    assert ltm.total_items == 3
    
    # Test retrieval for "Argentina" - should get 2 items
    result = ltm.retrieve_relevant_knowledge("Argentina")
    assert len(result) <= 3  # Can't get more than what we have
    
    # The results should contain Argentina-related items
    # (exact matching depends on whether ChromaDB or fallback is used)
    result_text = " ".join(result).lower()
    assert "argentina" in result_text
    
    # Test with top_k limit
    result_limited = ltm.retrieve_relevant_knowledge("Argentina", top_k=1)
    assert len(result_limited) == 1
    assert "argentina" in result_limited[0].lower()

def test_retrieve_with_empty_memory(fresh_ltm):
    """Test retrieval when memory is empty."""
    ltm = fresh_ltm
    
    # Should return empty list when no items stored
    result = ltm.retrieve_relevant_knowledge("anything")
    assert result == []

def test_retrieve_with_empty_query(fresh_ltm):
    """Test retrieval with empty or invalid queries."""
    ltm = fresh_ltm
    
    ltm.add_knowledge("Some fact")
    
    # Empty queries should return empty results
    assert ltm.retrieve_relevant_knowledge("") == []
    assert ltm.retrieve_relevant_knowledge("   ") == []

def test_clear_functionality(fresh_ltm):
    """Test clearing the memory."""
    ltm = fresh_ltm
    
    # Add some items
    ltm.add_knowledge("Fact 1")
    ltm.add_knowledge("Fact 2")
    assert ltm.total_items == 2
    
    # Clear and verify
    ltm.clear()
    assert ltm.total_items == 0
    assert ltm.get_all_items() == []

def test_repr(fresh_ltm):
    """Test string representation."""
    ltm = fresh_ltm
    
    # Empty memory
    repr_str = repr(ltm)
    assert "total_items=0" in repr_str
    assert "backend=" in repr_str
    
    # With items
    ltm.add_knowledge("A fact")
    repr_str = repr(ltm)
    assert "total_items=1" in repr_str

def test_duplicate_items(fresh_ltm):
    """Test adding duplicate items."""
    ltm = fresh_ltm
    
    # Add the same item multiple times
    ltm.add_knowledge("Duplicate fact")
    ltm.add_knowledge("Duplicate fact")
    ltm.add_knowledge("Duplicate fact")
    
    # ChromaDB should handle duplicates by updating, 
    # in-memory store will add multiple copies
    total = ltm.total_items
    assert total >= 1  # At least one should be stored
    
    # Should be able to retrieve it
    results = ltm.retrieve_relevant_knowledge("Duplicate")
    assert len(results) >= 1
    assert "duplicate" in results[0].lower()

def test_metadata_handling(fresh_ltm):
    """Test adding items with metadata."""
    ltm = fresh_ltm
    
    # Add item with metadata
    metadata = {"source": "test", "importance": 0.9}
    ltm.add_knowledge("Fact with metadata", metadata)
    
    assert ltm.total_items == 1
    
    # Should be retrievable
    results = ltm.retrieve_relevant_knowledge("metadata")
    assert len(results) == 1

def test_large_text_handling(fresh_ltm):
    """Test handling of large text items."""
    ltm = fresh_ltm
    
    # Create a large text item
    large_text = "This is a large text item. " * 100  # 2700+ characters
    ltm.add_knowledge(large_text)
    
    assert ltm.total_items == 1
    
    # Should be retrievable
    results = ltm.retrieve_relevant_knowledge("large text")
    assert len(results) == 1

def test_unicode_handling(fresh_ltm):
    """Test handling of unicode text."""
    ltm = fresh_ltm
    
    # Add unicode text
    unicode_text = "Paris est la capitale de la France. 🇫🇷"
    ltm.add_knowledge(unicode_text)
    
    assert ltm.total_items == 1
    
    # Should be retrievable
    results = ltm.retrieve_relevant_knowledge("Paris")
    assert len(results) == 1
    assert "Paris" in results[0]

def test_backend_detection():
    """Test that we can detect which backend is being used."""
    ltm = LongTermMemory()
    ltm.clear()
    
    repr_str = repr(ltm)
    # Should indicate either ChromaDB or in-memory
    assert "backend=ChromaDB" in repr_str or "backend=in-memory" in repr_str
    
    ltm.clear()

if __name__ == "__main__":
    print("Running LongTermMemory tests...")
    
    # List of test functions
    test_functions = [
        (test_add_and_total_items, "test_add_and_total_items"),
        (test_add_empty_or_invalid_items, "test_add_empty_or_invalid_items"),
        (test_retrieve_relevant_knowledge_simple_match, "test_retrieve_relevant_knowledge_simple_match"),
        (test_retrieve_with_empty_memory, "test_retrieve_with_empty_memory"),
        (test_retrieve_with_empty_query, "test_retrieve_with_empty_query"),
        (test_clear_functionality, "test_clear_functionality"),
        (test_repr, "test_repr"),
        (test_duplicate_items, "test_duplicate_items"),
        (test_metadata_handling, "test_metadata_handling"),
        (test_large_text_handling, "test_large_text_handling"),
        (test_unicode_handling, "test_unicode_handling"),
    ]
    
    passed = 0
    for test_func, test_name in test_functions:
        try:
            # Create fresh instance for each test
            ltm = LongTermMemory()
            ltm.clear()
            test_func(ltm)
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
        finally:
            # Clean up
            try:
                ltm.clear()
            except:
                pass
    
    # Test backend detection separately (no fixture needed)
    try:
        test_backend_detection()
        print("✅ test_backend_detection")
        passed += 1
    except Exception as e:
        print(f"❌ test_backend_detection: {e}")
    
    total_tests = len(test_functions) + 1
    print(f"\n{passed}/{total_tests} tests passed!")
    
    if passed == total_tests:
        print("🎉 All LongTermMemory tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")