#!/usr/bin/env python3
"""
benchmarks/fifa_scenario.py

Sets up the multi-hop question answering scenario from Section 8.1 of the paper.
This benchmark tests the system's ability to handle complex, multi-step reasoning tasks.
"""

import numpy as np
from typing import List, Tuple

# Add the project root to the path to allow importing from 'src'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.agent_pool import Agent, AgentPool
from src.core.task_graph import TaskGraph

# Constants for capability vector indices (makes agent/task config readable)
CAP_SPORTS, CAP_RETRIEVAL, CAP_ANALYTICS = 0, 1, 2

def setup_fifa_scenario() -> Tuple[AgentPool, TaskGraph]:
    """
    Creates the agents and 3-hop task graph for the FIFA World Cup query:
    "What is the GDP per capita of the country that won the most recent FIFA World Cup?"

    Returns:
        Tuple of configured (AgentPool, TaskGraph).
    """
    # --- Create agents ---
    agent_pool = AgentPool()
    agent_pool.add_agent(Agent(
        id="agent_sports_expert",
        capabilities=np.array([0.9, 0.2, 0.1])
    ))
    agent_pool.add_agent(Agent(
        id="agent_retrieval_specialist",
        capabilities=np.array([0.1, 0.9, 0.2])
    ))
    agent_pool.add_agent(Agent(
        id="agent_gdp_analyzer",
        capabilities=np.array([0.1, 0.3, 0.9])
    ))
    agent_pool.add_agent(Agent(
        id="agent_generalist",
        capabilities=np.array([0.5, 0.5, 0.5])
    ))

    # --- Create the multi-hop task graph ---
    task_graph = TaskGraph()
    task_graph.add_subtask(
        "t1_identify_event",
        required_capabilities=np.eye(3)[CAP_SPORTS],  # [1.0, 0.0, 0.0]
        description="Identify the winner of the most recent FIFA World Cup."
    )
    task_graph.add_subtask(
        "t2_retrieve_country",
        required_capabilities=np.eye(3)[CAP_RETRIEVAL],  # [0.0, 1.0, 0.0]
        description="Get the country name from the event result."
    )
    task_graph.add_subtask(
        "t3_calculate_gdp",
        required_capabilities=np.eye(3)[CAP_ANALYTICS],  # [0.0, 0.0, 1.0]
        description="Find the GDP per capita for the retrieved country."
    )
    task_graph.add_dependency("t1_identify_event", "t2_retrieve_country")
    task_graph.add_dependency("t2_retrieve_country", "t3_calculate_gdp")

    return agent_pool, task_graph

def main():
    """Demonstrates setting up the FIFA benchmark scenario."""
    print("====== Benchmarks: FIFA Scenario Setup Demo ======")
    agent_pool, task_graph = setup_fifa_scenario()

    print("\n--- Generated Agents ---")
    for agent in agent_pool.list_agents():
        print(f"  - {agent}")

    print("\n--- Generated Task Graph ---")
    print(f"Subtasks: {task_graph.get_all_subtasks()}")
    print(f"Dependencies: {list(task_graph.graph.edges())}")

    print("\nThis setup can now be passed to the main system orchestrator")
    print("to run a full benchmark of the coordination process.")

    print("\n=======================================================")
    print("✅ fifa_scenario.py executed successfully!")

if __name__ == "__main__":
    main()
