#!/usr/bin/env python3
"""
src/core/agent_pool.py

Defines the AgentPool class, which manages a collection of agents and implements
the match score function (Definition 3.1, Hybrid AI Brain paper). This class
supports dynamic registration, querying, and optimal matching of agents to tasks
based on capability vectors and assignment logic.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict

import logging

logger = logging.getLogger("hybrid_ai_brain.agent_pool")


@dataclass
class Agent:
    """
    Represents a single micro-cell agent in the swarm.
    """

    id: str
    capabilities: np.ndarray
    history: List[Any] = field(default_factory=list)
    _load: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if not isinstance(self.load, (int, float)) or not (0.0 <= self.load <= 1.0):
            raise ValueError(
                f"Agent load must be between 0.0 and 1.0, but got {self.load}."
            )
        # Validate capabilities
        self.capabilities = np.array(self.capabilities, dtype=float)
        norm = np.linalg.norm(self.capabilities)
        if norm > 0:
            self.capabilities = self.capabilities / norm
        else:
            raise ValueError("Agent capabilities vector must not be zero.")

    @property
    def load(self) -> float:
        return self._load

    @load.setter
    def load(self, value: float):
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            raise ValueError("Agent load must be a float in [0, 1].")
        self._load = float(value)

    def update_load(self, new_load: float):
        self.load = new_load
        logger.info(f"Agent '{self.id}': Load updated to {self.load:.2f}.")

    def add_history(self, entry: Any):
        self.history.append(entry)

    def __repr__(self) -> str:
        caps_str = np.array2string(self.capabilities, precision=2, floatmode="fixed")
        return (
            f"Agent(id='{self.id}', load={self.load:.2f}, " f"capabilities={caps_str})"
        )


class AgentPool:
    """
    A container to manage the swarm of all active micro-cell agents.
    """

    def __init__(self):
        self.agents: List[Agent] = []
        self._agent_map: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        if agent.id in self._agent_map:
            raise ValueError(f"Agent with ID '{agent.id}' already exists.")
        self.agents.append(agent)
        self._agent_map[agent.id] = agent
        logger.info(f"AgentPool: Added agent '{agent.id}'.")

    def add_agents(self, agent_list: List[Agent]):
        for agent in agent_list:
            self.add_agent(agent)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agent_map.get(agent_id)

    def list_agents(self) -> List[Agent]:
        return list(self.agents)

    def filter_by_load(self, max_load: float) -> List[Agent]:
        """Return agents with load less than or equal to max_load."""
        return [a for a in self.agents if a.load <= max_load]

    @property
    def count(self) -> int:
        return len(self.agents)

    def as_dataframe(self):
        try:
            import pandas as pd

            data = [
                {
                    "id": a.id,
                    "load": a.load,
                    **{f"cap_{i}": v for i, v in enumerate(a.capabilities)},
                }
                for a in self.agents
            ]
            return pd.DataFrame(data)
        except ImportError:
            logger.warning("Pandas not installed. DataFrame export unavailable.")
            return None

    def __repr__(self) -> str:
        return f"AgentPool(count={self.count})"

    def find_best_agent(self, required_capabilities: np.ndarray) -> Optional[Agent]:
        """
        Finds the agent with the highest dot-product score to the required capabilities.
        Returns None if there are no agents.
        """
        best = None
        best_score = float("-inf")
        req_norm = np.linalg.norm(required_capabilities)
        if req_norm == 0:
            raise ValueError("required_capabilities vector cannot be all zeros.")
        for agent in self.agents:
            score = np.dot(agent.capabilities, required_capabilities) / (
                np.linalg.norm(agent.capabilities) * req_norm
            )
            if score > best_score:
                best = agent
                best_score = score
        return best
