#!/usr/bin/env python3
"""
tests/test_consolidation.py

Comprehensive test with debug logging to identify consolidation issues
"""

import time
import logging
import pytest
import uuid # --- FIX: Import the uuid module ---
from unittest.mock import patch

# Import memory components here to be used throughout the file
from src.memory.working_memory import WorkingMemory
from src.memory.long_term_memory import LongTermMemory
from src.memory.flashbulb_buffer import FlashbulbBuffer
from src.memory.consolidation import ConsolidationProcess

# Setup detailed logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="function", autouse=True)
def reset_chromadb():
    """Ensure a clean state for ChromaDB before and after each test."""
    LongTermMemory.reset_global_state()
    yield
    LongTermMemory.reset_global_state()

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="consolidation_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_consolidation_components_individually(temp_storage_dir):
    """Test each consolidation component individually to isolate issues"""
    
    print("\n" + "="*50)
    print("COMPONENT TEST: Testing consolidation components individually")
    print("="*50)
    
    # Test 1: LongTermMemory direct functionality
    print("\n1. Testing LongTermMemory directly...")
    # --- FIX: Use the temp_storage_dir fixture and a unique collection name
    # to ensure this test is completely isolated from others. ---
    ltm = LongTermMemory(
        persist_directory=temp_storage_dir,
        collection_name=f"test_coll_isolated_{uuid.uuid4().hex[:8]}"
    )
    ltm.clear() 
    
    print(f"   Initial LTM items: {ltm.total_items}")
    assert ltm.total_items == 0, "LTM should start empty for an isolated test."

    ltm.add_knowledge("Test item 1")
    ltm.add_knowledge("Test item 2") 
    print(f"   After adding 2 items: {ltm.total_items}")
    assert ltm.total_items == 2, f"Expected 2 items, got {ltm.total_items}"
    
    results = ltm.retrieve_relevant_knowledge("test")
    print(f"   Retrieved {len(results)} items for 'test' query")
    assert len(results) == 2, "Should retrieve the two added items"
    
    # Test 2: Working Memory functionality
    print("\n2. Testing WorkingMemory...")
    wm = WorkingMemory(capacity=3)
    wm.add_item("test1", "value1")
    wm.add_item("test2", "value2")
    print(f"   WM size: {wm.current_size}")
    assert wm.current_size == 2
    
    # Test 3: FlashbulbBuffer functionality  
    print("\n3. Testing FlashbulbBuffer...")
    fb = FlashbulbBuffer(capacity=3)
    fb.capture_event("Test event 1", 0.9)
    items = fb.get_all_items()
    print(f"   FB items: {len(items)}")
    assert len(items) == 1
    
    # Test 4: ConsolidationProcess timing
    print("\n4. Testing ConsolidationProcess timing...")
    # Use the LTM instance we already verified is working correctly
    consolidator = ConsolidationProcess(wm, ltm, fb)
    
    # Manually set time to a past value to force consolidation
    consolidator.last_consolidation_time = time.time() - 5
    
    should_consolidate_now = consolidator.run_cycle_if_needed()
    print(f"   Consolidation cycle ran: {should_consolidate_now}")
    assert should_consolidate_now is True, "Consolidation should have run"
    
    print("\n✅ All component tests passed!")
