import numpy as np
import pytest

from src.core.agent_pool import Agent, AgentPool

def test_add_and_list_agents():
    pool = AgentPool()
    agent = Agent(id="agent1", capabilities=np.array([1.0, 0.0]))
    pool.add_agent(agent)
    agents = pool.list_agents()
    assert len(agents) == 1
    assert agents[0].id == "agent1"

def test_find_best_agent():
    pool = AgentPool()
    pool.add_agent(Agent(id="sports", capabilities=np.array([0.8, 0.1])))
    pool.add_agent(Agent(id="retriever", capabilities=np.array([0.2, 0.9])))
    # Task requires high retrieval skill
    best = pool.find_best_agent(np.array([0.1, 1.0]))
    assert best.id == "retriever"

def test_count_property():
    pool = AgentPool()
    for i in range(3):
        pool.add_agent(Agent(id=f"a{i}", capabilities=np.array([i, i])))
    assert pool.count == 3

