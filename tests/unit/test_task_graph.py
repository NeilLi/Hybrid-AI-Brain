import numpy as np
import pytest

from src.core.task_graph import TaskGraph

def test_add_subtask_and_retrieve():
    tg = TaskGraph()
    tg.add_subtask("t1", np.array([1.0, 0.0, 0.0]), description="Test subtask")
    assert "t1" in tg.get_all_subtasks()
    assert tg.get_subtask("t1")['description'] == "Test subtask"

def test_add_dependency_and_topological_sort():
    tg = TaskGraph()
    tg.add_subtask("t1", np.array([1.0]))
    tg.add_subtask("t2", np.array([0.5]))
    tg.add_dependency("t1", "t2")
    # t1 should come before t2 in topological sort
    order = tg.topological_sort()
    assert order.index("t1") < order.index("t2")

def test_cycle_detection():
    tg = TaskGraph()
    tg.add_subtask("a", np.array([0.]))
    tg.add_subtask("b", np.array([0.]))
    tg.add_dependency("a", "b")
    # Should not allow backward edge
    with pytest.raises(ValueError):
        tg.add_dependency("b", "a")

