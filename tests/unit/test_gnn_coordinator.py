import pytest
import numpy as np

from src.coordination.gnn_coordinator import GNNCoordinator

class DummyGraph:
    """A dummy graph placeholder for testing."""
    pass

def test_gnncoordinator_init_valid():
    # Should initialize with valid spectral norm bound
    gnn = GNNCoordinator(spectral_norm_bound=0.8, temperature=1.5)
    assert gnn.l_total_bound == 0.8
    assert gnn.temperature == 1.5

def test_gnncoordinator_init_invalid():
    # Should raise if spectral norm bound not in (0,1)
    with pytest.raises(ValueError):
        GNNCoordinator(spectral_norm_bound=1.1)
    with pytest.raises(ValueError):
        GNNCoordinator(spectral_norm_bound=0.0)

def test_run_message_passing_output_shape():
    gnn = GNNCoordinator()
    dummy_graph = DummyGraph()
    dummy_edge_features = {"t1": 1}
    embeddings = gnn._run_message_passing(dummy_graph, dummy_edge_features)
    # Check correct keys and shapes
    assert set(embeddings.keys()) >= {"task_1", "agent_A", "agent_B"}
    for v in embeddings.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (64,)

def test_assign_tasks_returns_assignment():
    gnn = GNNCoordinator()
    dummy_graph = DummyGraph()
    dummy_edge_features = {"t1": 1}
    assignments = gnn.assign_tasks(dummy_graph, dummy_edge_features)
    assert isinstance(assignments, dict)
    assert "task_1_fetch_data" in assignments
    assert assignments["task_1_fetch_data"] in {"agent_1_nlp", "agent_2_data"}


