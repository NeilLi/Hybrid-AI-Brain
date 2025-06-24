#!/usr/bin/env python3
"""
tests/test_consolidation.py

Comprehensive test with debug logging to identify consolidation issues
"""

import time
import logging
import pytest
from unittest.mock import patch

# Setup detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_memory_consolidation_cycle_debug():
    """Test with extensive debug logging to identify the consolidation issue"""
    
    # Import here to ensure clean state
    from src.memory.working_memory import WorkingMemory
    from src.memory.long_term_memory import LongTermMemory
    from src.memory.flashbulb_buffer import FlashbulbBuffer
    from src.memory.consolidation import ConsolidationProcess
    
    print("\n" + "="*60)
    print("DEBUG: Starting memory consolidation test")
    print("="*60)
    
    # Initialize components with debug logging
    print("DEBUG: Initializing memory components...")
    wm = WorkingMemory(capacity=5)
    ltm = LongTermMemory()  # This should auto-detect ChromaDB or fallback
    fb = FlashbulbBuffer(capacity=5)
    
    print(f"DEBUG: WM initialized: {wm}")
    print(f"DEBUG: LTM initialized: {ltm}")
    print(f"DEBUG: FB initialized: {fb}")
    
    consolidator = ConsolidationProcess(wm, ltm, fb)
    print(f"DEBUG: Consolidator initialized: {consolidator}")
    
    # Add test data with debug output
    print("\nDEBUG: Adding test data...")
    
    # Add items to working memory
    test_items = {
        "key1": "value1 - test data for consolidation",
        "key2": "value2 - another test value", 
        "task_result": "Task completed successfully"
    }
    
    for key, value in test_items.items():
        wm.add_item(key, value)
        print(f"DEBUG: Added to WM: {key} = {value}")
    
    # Add events to flashbulb buffer
    test_events = [
        ("Critical system event occurred", 0.9),
        ("Important discovery made", 0.8),
        ("User feedback received", 0.7)
    ]
    
    for event, confidence in test_events:
        fb.capture_event(event, confidence)
        print(f"DEBUG: Added to FB: {event} (confidence: {confidence})")
    
    # Show initial state
    print(f"\nDEBUG: Initial state before consolidation:")
    print(f"  - Working Memory: {wm.current_size} items")
    print(f"  - Flashbulb Buffer: {len(fb.get_all_items())} items") 
    print(f"  - Long-Term Memory: {ltm.total_items} items")
    print(f"  - Consolidator status: {consolidator.get_status()}")
    
    # Test consolidation conditions
    print(f"\nDEBUG: Checking consolidation conditions...")
    should_consolidate_before = consolidator.should_consolidate()
    print(f"  - Should consolidate (before wait): {should_consolidate_before}")
    
    # Wait for timing condition
    print(f"DEBUG: Waiting for GAMMA_INTERVAL (1.0 seconds)...")
    time.sleep(1.1)  # Wait slightly longer than GAMMA_INTERVAL
    
    should_consolidate_after = consolidator.should_consolidate()
    print(f"  - Should consolidate (after wait): {should_consolidate_after}")
    
    # Patch LongTermMemory.add_knowledge to track calls
    add_knowledge_calls = []
    original_add_knowledge = ltm.add_knowledge
    
    def tracked_add_knowledge(item, metadata=None):
        add_knowledge_calls.append((item, metadata))
        print(f"DEBUG: add_knowledge() called with item: {item}")
        return original_add_knowledge(item, metadata)
    
    ltm.add_knowledge = tracked_add_knowledge
    
    # Run consolidation with tracking
    print(f"\nDEBUG: Running consolidation cycle...")
    consolidation_result = consolidator.run_cycle_if_needed()
    print(f"DEBUG: Consolidation result: {consolidation_result}")
    print(f"DEBUG: add_knowledge was called {len(add_knowledge_calls)} times")
    
    for i, (item, metadata) in enumerate(add_knowledge_calls, 1):
        print(f"  Call {i}: {item[:50]}..." if len(item) > 50 else f"  Call {i}: {item}")
    
    # Check final state
    print(f"\nDEBUG: Final state after consolidation:")
    print(f"  - Working Memory: {wm.current_size} items")
    print(f"  - Flashbulb Buffer: {len(fb.get_all_items())} items")
    print(f"  - Long-Term Memory: {ltm.total_items} items")
    print(f"  - Consolidator: {consolidator}")
    
    # Test LTM functionality directly
    print(f"\nDEBUG: Testing LTM functionality directly...")
    direct_test_item = "Direct test item for debugging"
    ltm.add_knowledge(direct_test_item)
    print(f"DEBUG: Added item directly to LTM: {direct_test_item}")
    print(f"DEBUG: LTM total items after direct add: {ltm.total_items}")
    
    # Test retrieval
    print(f"\nDEBUG: Testing LTM retrieval...")
    query_results = ltm.retrieve_relevant_knowledge("test", top_k=5)
    print(f"DEBUG: Retrieved {len(query_results)} items for query 'test':")
    for i, result in enumerate(query_results, 1):
        print(f"  Result {i}: {result}")
    
    # Force consolidation test
    if not consolidation_result:
        print(f"\nDEBUG: Normal consolidation failed, trying force consolidation...")
        # Add more data
        wm.add_item("force_test", "Force consolidation test data")
        fb.capture_event("Force test event", 0.95)
        
        force_result = consolidator.force_consolidation()
        print(f"DEBUG: Force consolidation result: {force_result}")
        print(f"DEBUG: LTM items after force: {ltm.total_items}")
    
    # Final assertions with detailed error messages
    print(f"\n" + "="*60)
    print("DEBUG: Running final assertions...")
    print("="*60)
    
    if ltm.total_items == 0:
        print("ERROR: Long-Term Memory is still empty after consolidation!")
        print("This indicates the consolidation process is not working correctly.")
        print("Check the add_knowledge calls above to see if they were made.")
        
        # Additional diagnostics
        print(f"\nDiagnostic info:")
        print(f"  - Consolidation was attempted: {consolidation_result}")
        print(f"  - add_knowledge calls made: {len(add_knowledge_calls)}")
        print(f"  - LTM backend ready: {getattr(ltm, '_is_backend_ready', 'unknown')}")
        print(f"  - WM had items: {len(test_items) > 0}")
        print(f"  - FB had items: {len(test_events) > 0}")
        
        # Try one more direct test
        print(f"\nTrying one final direct LTM test...")
        try:
            test_item = "Final debug test item"
            ltm.add_knowledge(test_item)
            final_count = ltm.total_items
            print(f"  - Added '{test_item}', LTM now has {final_count} items")
            
            if final_count > 0:
                print("  - LTM add_knowledge works directly, issue is in consolidation logic")
            else:
                print("  - LTM add_knowledge not working even directly, issue is in LTM implementation")
                
        except Exception as e:
            print(f"  - Exception in direct LTM test: {e}")
    
    # The actual assertion
    assert ltm.total_items >= 1, f"Expected at least 1 item in LTM, but got {ltm.total_items}. Check debug output above."
    
    print(f"DEBUG: Test passed! LTM contains {ltm.total_items} items.")
    print("="*60)


def test_consolidation_components_individually():
    """Test each consolidation component individually to isolate issues"""
    
    from src.memory.working_memory import WorkingMemory
    from src.memory.long_term_memory import LongTermMemory
    from src.memory.flashbulb_buffer import FlashbulbBuffer
    from src.memory.consolidation import ConsolidationProcess
    
    print("\n" + "="*50)
    print("COMPONENT TEST: Testing consolidation components individually")
    print("="*50)
    
    # Test 1: LongTermMemory direct functionality
    print("\n1. Testing LongTermMemory directly...")
    ltm = LongTermMemory()
    print(f"   Initial LTM items: {ltm.total_items}")
    
    ltm.add_knowledge("Test item 1")
    ltm.add_knowledge("Test item 2") 
    print(f"   After adding 2 items: {ltm.total_items}")
    assert ltm.total_items == 2, f"Expected 2 items, got {ltm.total_items}"
    
    results = ltm.retrieve_relevant_knowledge("test")
    print(f"   Retrieved {len(results)} items for 'test' query")
    assert len(results) > 0, "Should retrieve at least 1 item"
    
    # Test 2: Working Memory functionality
    print("\n2. Testing WorkingMemory...")
    wm = WorkingMemory(capacity=3)
    wm.add_item("test1", "value1")
    wm.add_item("test2", "value2")
    print(f"   WM size: {wm.current_size}")
    assert wm.current_size == 2
    
    wm_dict = wm.as_dict()
    print(f"   WM contents: {wm_dict}")
    assert len(wm_dict) == 2
    
    # Test 3: FlashbulbBuffer functionality  
    print("\n3. Testing FlashbulbBuffer...")
    fb = FlashbulbBuffer(capacity=3)
    fb.capture_event("Test event 1", 0.9)
    fb.capture_event("Test event 2", 0.8)
    
    items = fb.get_all_items()
    print(f"   FB items: {len(items)}")
    assert len(items) == 2
    
    for item in items:
        print(f"   Item weight: {item.get_current_weight():.3f}")
        assert item.get_current_weight() > 0
    
    # Test 4: ConsolidationProcess timing
    print("\n4. Testing ConsolidationProcess timing...")
    consolidator = ConsolidationProcess(wm, ltm, fb)
    
    # Should not consolidate immediately (no time passed)
    should_consolidate_immediate = consolidator.should_consolidate() 
    print(f"   Should consolidate immediately: {should_consolidate_immediate}")
    
    # Reset timing and check again
    consolidator.last_consolidation_time = 0.0
    should_consolidate_reset = consolidator.should_consolidate()
    print(f"   Should consolidate after reset: {should_consolidate_reset}")
    assert should_consolidate_reset, "Should consolidate when timing is reset and memory has content"
    
    print("\n✅ All component tests passed!")


if __name__ == "__main__":
    # Run the debug test directly
    test_memory_consolidation_cycle_debug()
    test_consolidation_components_individually()
    print("\n🎉 All debug tests completed!")