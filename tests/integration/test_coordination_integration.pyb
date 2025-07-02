# In tests/integration/test_coordination_integration.py

import numpy as np
import pytest

from src.core.agent_pool import Agent, AgentPool
from src.core.task_graph import TaskGraph
from src.coordination.gnn_coordinator import GNNCoordinator

@pytest.fixture
def agent_pool_and_task_graph():
    pool = AgentPool()
    pool.add_agent(Agent(id="agent1", capabilities=np.array([1.0, 0.0, 0.0])))
    pool.add_agent(Agent(id="agent2", capabilities=np.array([0.0, 1.0, 0.0])))

    tg = TaskGraph()
    tg.add_subtask("task_a", required_capabilities=np.array([1.0, 0.0, 0.0]))
    tg.add_subtask("task_b", required_capabilities=np.array([0.0, 1.0, 0.0]))
    tg.add_dependency("task_a", "task_b")
    return pool, tg

def test_gnn_coordinator_assignment(agent_pool_and_task_graph):
    pool, tg = agent_pool_and_task_graph
    
    # --- CORRECTED: Use the public methods from your actual classes ---
    tasks_to_assign = tg.get_all_subtasks()
    available_agents = [agent.id for agent in pool.list_agents()]
    
    # Create a mock bio_signals dictionary for the test
    mock_bio_signals = {
        "pso_global_best": np.random.rand(64),
        "pheromone_levels": {},
        "conflict_weights": (0.5, 0.5)
    }

    coordinator = GNNCoordinator()
    
    assignments = coordinator.assign_tasks(
        tasks=tasks_to_assign, 
        agents=available_agents, 
        bio_signals=mock_bio_signals
    )
    
    assert isinstance(assignments, dict)
    assert set(assignments.keys()) == set(tasks_to_assign)
    for agent_id in assignments.values():
        assert agent_id in available_agents