#!/usr/bin/env python3
"""
tests/unit/test_task_graph.py

Fixed tests for TaskGraph that handle the capability vector requirements properly.
"""

import numpy as np
import pytest
from src.core.task_graph import TaskGraph

def test_add_subtask_and_retrieve():
    """Test adding a subtask and retrieving its information."""
    tg = TaskGraph()
    capabilities = np.array([1.0, 0.0, 0.0])
    
    tg.add_subtask("t1", capabilities, description="Test subtask")
    
    # Check that subtask was added
    assert "t1" in tg.get_all_subtasks()
    
    # Check that we can retrieve the subtask with its attributes
    subtask_info = tg.get_subtask("t1")
    assert subtask_info['description'] == "Test subtask"
    
    # Check that capabilities were normalized
    assert 'required_capabilities' in subtask_info
    retrieved_caps = subtask_info['required_capabilities']
    assert isinstance(retrieved_caps, np.ndarray)
    # Should be normalized to unit length
    assert abs(np.linalg.norm(retrieved_caps) - 1.0) < 1e-6

def test_add_subtask_capability_normalization():
    """Test that capability vectors are properly normalized."""
    tg = TaskGraph()
    
    # Test with non-unit vector
    capabilities = np.array([3.0, 4.0])  # magnitude = 5.0
    tg.add_subtask("t1", capabilities)
    
    retrieved_caps = tg.get_subtask("t1")['required_capabilities']
    expected_normalized = np.array([0.6, 0.8])  # [3/5, 4/5]
    
    assert np.allclose(retrieved_caps, expected_normalized)
    assert abs(np.linalg.norm(retrieved_caps) - 1.0) < 1e-6

def test_add_subtask_zero_vector_error():
    """Test that zero capability vectors raise an error."""
    tg = TaskGraph()
    
    with pytest.raises(ValueError, match="required_capabilities vector cannot be all zeros"):
        tg.add_subtask("t1", np.array([0.0, 0.0, 0.0]))
    
    with pytest.raises(ValueError, match="required_capabilities vector cannot be all zeros"):
        tg.add_subtask("t2", np.array([]))  # empty array
    
    with pytest.raises(ValueError, match="required_capabilities vector cannot be all zeros"):
        tg.add_subtask("t3", np.array([0.0]))  # single zero

def test_add_dependency_and_topological_sort():
    """Test adding dependencies and verifying topological order."""
    tg = TaskGraph()
    
    # Add subtasks with non-zero capabilities
    tg.add_subtask("t1", np.array([1.0, 0.0]))
    tg.add_subtask("t2", np.array([0.0, 1.0]))
    
    # Add dependency: t1 must complete before t2
    tg.add_dependency("t1", "t2")
    
    # Check that dependency was added
    dependencies = tg.get_dependencies()
    assert ("t1", "t2") in dependencies
    
    # Verify topological order: t1 should come before t2
    order = tg.topological_sort()
    assert order.index("t1") < order.index("t2")

def test_add_dependency_with_attributes():
    """Test adding dependencies with cost and risk attributes."""
    tg = TaskGraph()
    
    tg.add_subtask("t1", np.array([1.0]))
    tg.add_subtask("t2", np.array([0.5]))
    
    # Add dependency with attributes
    tg.add_dependency("t1", "t2", cost=2.5, risk=0.3)
    
    # Check edge attributes
    edge_attrs = tg.get_edge_attributes("t1", "t2")
    assert edge_attrs['cost'] == 2.5
    assert edge_attrs['risk'] == 0.3

def test_cycle_detection():
    """Test that cycles are properly detected and prevented."""
    tg = TaskGraph()
    
    # Add subtasks with valid (non-zero) capabilities
    tg.add_subtask("a", np.array([1.0]))
    tg.add_subtask("b", np.array([0.5]))
    
    # Add first dependency: a -> b
    tg.add_dependency("a", "b")
    
    # Attempting to add reverse dependency should raise ValueError (creates cycle)
    with pytest.raises(ValueError, match="would create a cycle"):
        tg.add_dependency("b", "a")

def test_cycle_detection_longer_cycle():
    """Test cycle detection with longer dependency chains."""
    tg = TaskGraph()
    
    # Create a longer chain: a -> b -> c
    tg.add_subtask("a", np.array([1.0]))
    tg.add_subtask("b", np.array([0.5]))
    tg.add_subtask("c", np.array([0.3]))
    
    tg.add_dependency("a", "b")
    tg.add_dependency("b", "c")
    
    # Trying to close the loop should fail
    with pytest.raises(ValueError, match="would create a cycle"):
        tg.add_dependency("c", "a")

def test_dependency_on_nonexistent_task():
    """Test that dependencies on non-existent tasks raise errors."""
    tg = TaskGraph()
    
    tg.add_subtask("existing", np.array([1.0]))
    
    # Should raise error when trying to add dependency with non-existent tasks
    with pytest.raises(ValueError, match="Both tasks must exist"):
        tg.add_dependency("existing", "nonexistent")
    
    with pytest.raises(ValueError, match="Both tasks must exist"):
        tg.add_dependency("nonexistent", "existing")

def test_get_nonexistent_subtask():
    """Test that getting a non-existent subtask raises KeyError."""
    tg = TaskGraph()
    
    with pytest.raises(KeyError, match="Task 'nonexistent' not found"):
        tg.get_subtask("nonexistent")

def test_get_nonexistent_edge_attributes():
    """Test that getting attributes of non-existent edge raises KeyError."""
    tg = TaskGraph()
    
    tg.add_subtask("t1", np.array([1.0]))
    tg.add_subtask("t2", np.array([0.5]))
    # Don't add the edge
    
    with pytest.raises(KeyError, match="No dependency from 't1' to 't2'"):
        tg.get_edge_attributes("t1", "t2")

def test_bulk_operations():
    """Test bulk addition of subtasks and dependencies."""
    tg = TaskGraph()
    
    # Bulk add subtasks
    tasks = {
        "task1": np.array([1.0, 0.0]),
        "task2": np.array([0.0, 1.0]),
        "task3": np.array([0.5, 0.5])
    }
    tg.add_subtasks_bulk(tasks)
    
    assert len(tg.get_all_subtasks()) == 3
    assert "task1" in tg.get_all_subtasks()
    assert "task2" in tg.get_all_subtasks()
    assert "task3" in tg.get_all_subtasks()
    
    # Bulk add dependencies
    dependencies = [
        ("task1", "task2", 1.0, 0.1),
        ("task2", "task3", 2.0, 0.2)
    ]
    tg.add_dependencies_bulk(dependencies)
    
    deps = tg.get_dependencies()
    assert ("task1", "task2") in deps
    assert ("task2", "task3") in deps

def test_serialization():
    """Test converting TaskGraph to/from dictionary representation."""
    tg = TaskGraph()
    
    # Add some data
    tg.add_subtask("t1", np.array([1.0, 0.0]), description="First task")
    tg.add_subtask("t2", np.array([0.0, 1.0]), description="Second task")
    tg.add_dependency("t1", "t2", cost=1.5, risk=0.2)
    
    # Convert to dict
    graph_dict = tg.to_dict()
    
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    assert len(graph_dict["nodes"]) == 2
    assert len(graph_dict["edges"]) == 1
    
    # Check node data
    node_ids = [node["id"] for node in graph_dict["nodes"]]
    assert "t1" in node_ids
    assert "t2" in node_ids
    
    # Check edge data
    edge = graph_dict["edges"][0]
    assert edge["from"] == "t1"
    assert edge["to"] == "t2"
    assert edge["cost"] == 1.5
    assert edge["risk"] == 0.2
    
    # Test round-trip: dict -> TaskGraph
    tg_restored = TaskGraph.from_dict(graph_dict)
    
    assert len(tg_restored.get_all_subtasks()) == 2
    assert tg_restored.get_subtask("t1")["description"] == "First task"
    assert tg_restored.get_edge_attributes("t1", "t2")["cost"] == 1.5

def test_topological_order_complex():
    """Test topological ordering with a more complex dependency graph."""
    tg = TaskGraph()
    
    # Create a diamond-shaped dependency graph
    #   A
    #  / \
    # B   C
    #  \ /
    #   D
    
    tg.add_subtask("A", np.array([1.0]))
    tg.add_subtask("B", np.array([0.5]))
    tg.add_subtask("C", np.array([0.3]))
    tg.add_subtask("D", np.array([0.7]))
    
    tg.add_dependency("A", "B")
    tg.add_dependency("A", "C")
    tg.add_dependency("B", "D")
    tg.add_dependency("C", "D")
    
    order = tg.topological_order()
    
    # A must come first
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    
    # B and C must come before D
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")

def test_repr():
    """Test string representation of TaskGraph."""
    tg = TaskGraph()
    
    # Empty graph
    repr_str = repr(tg)
    assert "TaskGraph" in repr_str
    assert "nodes=[]" in repr_str
    assert "dependencies=[]" in repr_str
    
    # Graph with data
    tg.add_subtask("t1", np.array([1.0]))
    tg.add_subtask("t2", np.array([0.5]))
    tg.add_dependency("t1", "t2")
    
    repr_str = repr(tg)
    assert "nodes=['t1', 't2']" in repr_str or "nodes=['t2', 't1']" in repr_str
    assert "dependencies=[('t1', 't2')]" in repr_str

if __name__ == "__main__":
    print("Running TaskGraph tests...")
    
    test_functions = [
        test_add_subtask_and_retrieve,
        test_add_subtask_capability_normalization,
        test_add_subtask_zero_vector_error,
        test_add_dependency_and_topological_sort,
        test_add_dependency_with_attributes,
        test_cycle_detection,
        test_cycle_detection_longer_cycle,
        test_dependency_on_nonexistent_task,
        test_get_nonexistent_subtask,
        test_get_nonexistent_edge_attributes,
        test_bulk_operations,
        test_serialization,
        test_topological_order_complex,
        test_repr,
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
        print("🎉 All TaskGraph tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")