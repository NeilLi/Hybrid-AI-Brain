#!/usr/bin/env python3
"""
benchmarks/fifa_scenario.py

Sets up the multi‑hop question‑answering scenario from Section 8.1 of the
paper.  This benchmark tests the system’s ability to handle complex,
multi‑step reasoning tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Add the project root to the path so we can import from ``src``
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.agent_pool import Agent, AgentPool  # type: ignore
from src.core.task_graph import TaskGraph  # type: ignore

# ---------------------------------------------------------------------------
# Capability indexing helpers (for readability only)
# ---------------------------------------------------------------------------

CAP_DOMAIN = 0      # Sports‑domain knowledge
CAP_RETRIEVAL = 1   # Document / fact retrieval
CAP_ANALYTICS = 2   # Numerical / economic analytics


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def setup_fifa_scenario() -> Tuple[AgentPool, TaskGraph]:
    """Build the agent pool and a 3‑hop task graph for the query:

    *“What is the GDP per capita of the country that won the most recent FIFA
    World Cup?”*
    """

    # ── 1 · Create agents ────────────────────────────────────────────────────
    agent_pool = AgentPool()

    agent_pool.add_agent(
        Agent(
            agent_id="agent_sports_expert",
            capabilities=np.array([0.9, 0.2, 0.1]),
        )
    )
    agent_pool.add_agent(
        Agent(
            agent_id="agent_retrieval_specialist",
            capabilities=np.array([0.1, 0.9, 0.2]),
        )
    )
    agent_pool.add_agent(
        Agent(
            agent_id="agent_gdp_analyzer",
            capabilities=np.array([0.1, 0.3, 0.9]),
        )
    )
    agent_pool.add_agent(
        Agent(
            agent_id="agent_generalist",
            capabilities=np.array([0.5, 0.5, 0.5]),
        )
    )

    # ── 2 · Create the task graph (3‑hop chain) ─────────────────────────────
    task_graph = TaskGraph()

    task_graph.add_subtask(
        "t1_identify_event",
        required_capabilities=np.eye(3)[CAP_DOMAIN],  # [1.0, 0.0, 0.0]
        description="Identify the winner of the most recent FIFA World Cup.",
    )
    task_graph.add_subtask(
        "t2_retrieve_country",
        required_capabilities=np.eye(3)[CAP_RETRIEVAL],  # [0.0, 1.0, 0.0]
        description="Retrieve the country name from the event result.",
    )
    task_graph.add_subtask(
        "t3_calculate_gdp",
        required_capabilities=np.eye(3)[CAP_ANALYTICS],  # [0.0, 0.0, 1.0]
        description="Lookup the GDP per capita for the retrieved country.",
    )

    # Define dependencies: t1 → t2 → t3
    task_graph.add_dependency("t1_identify_event", "t2_retrieve_country")
    task_graph.add_dependency("t2_retrieve_country", "t3_calculate_gdp")

    return agent_pool, task_graph


# ---------------------------------------------------------------------------
# Demo entry‑point – just for quick validation
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover  (CLI helper)
    print("====== Benchmarks: FIFA Scenario Setup Demo ======")
    agent_pool, task_graph = setup_fifa_scenario()

    print("\n--- Generated Agents ---")
    for agent in agent_pool.list_agents():
        print(f"  • {agent}")

    print("\n--- Generated Task Graph ---")
    print("Subtasks        :", task_graph.get_all_subtasks())
    print("Dependencies    :", list(task_graph.graph.edges()))

    print("\nThis setup can now be passed to the main orchestrator to run the full benchmark.")
    print("\n=======================================================")
    print("✅ fifa_scenario.py executed successfully!")


if __name__ == "__main__":
    main()
