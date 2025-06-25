#!/usr/bin/env python3
"""
tests/unit/test_long_term_memory.py

Updated tests for LongTermMemory with persistent storage support.
Fixed to handle ChromaDB singleton issues properly.
"""

import pytest
import time
import tempfile
import shutil
import uuid
from pathlib import Path
from src.memory.long_term_memory import LongTermMemory

@pytest.fixture(scope="function")
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    temp_dir = tempfile.mkdtemp(prefix="ltm_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function", autouse=True)
def reset_chromadb_state():
    """Reset ChromaDB global state before each test."""
    LongTermMemory.reset_global_state()
    yield
    LongTermMemory.reset_global_state()

@pytest.fixture
def fresh_ltm(temp_storage_dir):
    """Create a fresh LongTermMemory instance for each test with isolated storage."""
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    ltm = LongTermMemory(
        persist_directory=temp_storage_dir,
        collection_name=collection_name
    )
    ltm.clear()
    yield ltm

def test_add_and_total_items(fresh_ltm):
    """Test adding items and counting them."""
    assert fresh_ltm.total_items == 0
    fresh_ltm.add_knowledge("Fact 1")
    assert fresh_ltm.total_items == 1
    fresh_ltm.add_knowledge("Fact 2")
    assert fresh_ltm.total_items == 2
    all_items = fresh_ltm.get_all_items()
    assert len(all_items) == 2
    # --- FIX: Test now passes because get_all_items returns a list of strings ---
    assert "Fact 1" in all_items
    assert "Fact 2" in all_items

def test_persistence_across_instances(temp_storage_dir):
    """Test that data persists across different LongTermMemory instances."""
    collection_name = f"persistence_test_{uuid.uuid4().hex[:8]}"
    
    ltm1 = LongTermMemory(persist_directory=temp_storage_dir, collection_name=collection_name)
    ltm1.clear()
    ltm1.add_knowledge("Persistent fact 1")
    assert ltm1.total_items == 1
    ltm1 = None
    
    ltm2 = LongTermMemory(persist_directory=temp_storage_dir, collection_name=collection_name)
    assert ltm2.total_items == 1
    all_items = ltm2.get_all_items()
    # --- FIX: Test now passes because get_all_items returns a list of strings ---
    assert "Persistent fact 1" in all_items
    ltm2.clear()

def test_retrieve_relevant_knowledge_semantic_search(fresh_ltm):
    """Test semantic retrieval functionality."""
    fresh_ltm.add_knowledge("Python is a popular programming language for AI.")
    assert fresh_ltm.total_items == 1
    
    programming_results = fresh_ltm.retrieve_relevant_knowledge("coding", top_k=1)
    assert len(programming_results) == 1
    assert "python" in programming_results[0].lower()

def test_metadata_handling_and_retrieval(fresh_ltm):
    """Test adding items with metadata and retrieving metadata."""
    ltm = fresh_ltm
    metadata = {"source": "test", "importance": 0.9}
    ltm.add_knowledge("Quantum physics is a fundamental theory.", metadata)
    assert ltm.total_items == 1

    if ltm._is_backend_ready:
        metadata_results = ltm.get_metadata_for_query("physics", top_k=1)
        assert len(metadata_results) == 1
        result = metadata_results[0]
        # --- FIX: Test now passes because the result is a dict with these keys ---
        assert "document" in result
        assert "metadata" in result
        
        result_metadata = result["metadata"]
        assert result_metadata.get("source") == "test"
        assert result_metadata.get("importance") == 0.9

