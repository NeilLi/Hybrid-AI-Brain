import numpy as np
import pytest

from src.core.agent_pool import Agent, AgentPool
from src.core.task_graph import TaskGraph
from src.coordination.gnn_coordinator import GNNCoordinator

@pytest.fixture
def agent_pool_and_task_graph():
    # Setup agents
    pool = AgentPool()
    pool.add_agent(Agent(id="agent1", capabilities=np.array([1.0, 0.0, 0.0])))
    pool.add_agent(Agent(id="agent2", capabilities=np.array([0.0, 1.0, 0.0])))

    # Setup task graph
    tg = TaskGraph()
    tg.add_subtask("task_a", required_capabilities=np.array([1.0, 0.0, 0.0]))
    tg.add_subtask("task_b", required_capabilities=np.array([0.0, 1.0, 0.0]))
    tg.add_dependency("task_a", "task_b")
    return pool, tg

def test_gnn_coordinator_assignment(agent_pool_and_task_graph):
    pool, tg = agent_pool_and_task_graph
    # Simulate agent-task graph (this is a placeholder for actual structure)
    agent_task_graph = {}  # In real code, build your bipartite graph structure

    # Edge features can be empty or minimal for this demo
    edge_features = {}

    coordinator = GNNCoordinator(spectral_norm_bound=0.7)
    assignments = coordinator.assign_tasks(agent_task_graph, edge_features)
    assert isinstance(assignments, dict)
    # (Optional) Check at least the keys/values structure
    for task_id, agent_id in assignments.items():
        assert isinstance(task_id, str)
        assert isinstance(agent_id, str)

