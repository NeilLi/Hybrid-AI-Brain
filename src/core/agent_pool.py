#!/usr/bin/env python3
"""
src/core/agent_pool.py

Enhanced AgentPool class that supports both dictionary and numpy array capabilities.
This maintains backward compatibility while enabling the theoretical framework
from the Hybrid AI Brain paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Union
import logging

logger = logging.getLogger("hybrid_ai_brain.agent_pool")


@dataclass
class Agent:
    """
    Represents a single micro-cell agent in the swarm.
    Enhanced to support both dictionary and numpy array capabilities.
    """

    agent_id: str  # Changed from 'id' to 'agent_id' for consistency
    capabilities: Union[Dict[str, float], np.ndarray]
    history: List[Any] = field(default_factory=list)
    _load: float = field(default=0.0, repr=False)
    
    # Additional fields for enhanced functionality
    role: str = field(default="worker", repr=False)  # For ABC roles if needed
    performance_history: List[float] = field(default_factory=list, repr=False)
    
    # Internal fields
    _capabilities_dict: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _capabilities_array: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _capability_names: Optional[List[str]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Validate and set load
        if not isinstance(self._load, (int, float)) or not (0.0 <= self._load <= 1.0):
            raise ValueError(
                f"Agent load must be between 0.0 and 1.0, but got {self._load}."
            )
        
        # Process capabilities based on input type
        self._process_capabilities()
        
        # Validate that we have valid capabilities
        if self._capabilities_array is None or len(self._capabilities_array) == 0:
            raise ValueError("Agent must have valid capabilities.")
        
        # Normalize capabilities array
        norm = np.linalg.norm(self._capabilities_array)
        if norm > 0:
            self._capabilities_array = self._capabilities_array / norm
        else:
            raise ValueError("Agent capabilities vector must not be zero.")
    
    def _process_capabilities(self):
        """Process capabilities input and create both dict and array representations."""
        if isinstance(self.capabilities, dict):
            # Input is dictionary - create both representations
            self._capabilities_dict = self.capabilities.copy()
            self._capability_names = list(self.capabilities.keys())
            self._capabilities_array = np.array(list(self.capabilities.values()), dtype=float)
            
        elif isinstance(self.capabilities, (list, tuple, np.ndarray)):
            # Input is array-like - create array and generic dict
            self._capabilities_array = np.array(self.capabilities, dtype=float)
            
            # Create a generic dictionary representation
            self._capability_names = [f"capability_{i}" for i in range(len(self._capabilities_array))]
            self._capabilities_dict = {
                name: float(value) 
                for name, value in zip(self._capability_names, self._capabilities_array)
            }
            
        else:
            raise TypeError(
                f"Capabilities must be dict, list, tuple, or numpy array, "
                f"got {type(self.capabilities)}"
            )
    
    @property
    def id(self) -> str:
        """Backward compatibility property."""
        return self.agent_id
    
    @property
    def load(self) -> float:
        return self._load

    @load.setter
    def load(self, value: float):
        if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
            raise ValueError("Agent load must be a float in [0, 1].")
        self._load = float(value)

    @property
    def capabilities_dict(self) -> Dict[str, float]:
        """Get capabilities as dictionary."""
        return self._capabilities_dict.copy()
    
    @property
    def capabilities_array(self) -> np.ndarray:
        """Get capabilities as numpy array."""
        return self._capabilities_array.copy()
    
    @property
    def capability_names(self) -> List[str]:
        """Get list of capability names."""
        return self._capability_names.copy()
    
    def get_capability(self, name: str) -> float:
        """Get a specific capability by name."""
        if name not in self._capabilities_dict:
            raise KeyError(f"Capability '{name}' not found. Available: {self._capability_names}")
        return self._capabilities_dict[name]
    
    def has_capability(self, name: str) -> bool:
        """Check if agent has a specific capability."""
        return name in self._capabilities_dict
    
    def update_capability(self, name: str, value: float):
        """Update a specific capability and renormalize."""
        if name not in self._capabilities_dict:
            raise KeyError(f"Capability '{name}' not found. Available: {self._capability_names}")
        
        # Update dictionary
        self._capabilities_dict[name] = float(value)
        
        # Update array
        idx = self._capability_names.index(name)
        self._capabilities_array[idx] = float(value)
        
        # Renormalize
        norm = np.linalg.norm(self._capabilities_array)
        if norm > 0:
            self._capabilities_array = self._capabilities_array / norm
            # Update dictionary with normalized values
            for i, cap_name in enumerate(self._capability_names):
                self._capabilities_dict[cap_name] = float(self._capabilities_array[i])
        
        logger.info(f"Agent '{self.agent_id}': Updated capability '{name}' to {value}")

    def update_load(self, new_load: float):
        self.load = new_load
        logger.info(f"Agent '{self.agent_id}': Load updated to {self.load:.2f}.")

    def add_history(self, entry: Any):
        self.history.append(entry)
    
    def add_performance_score(self, score: float):
        """Add a performance score to history."""
        self.performance_history.append(float(score))
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
    
    def get_average_performance(self) -> float:
        """Get average performance score."""
        if not self.performance_history:
            return 0.5  # Default neutral performance
        return np.mean(self.performance_history)

    def __repr__(self) -> str:
        if len(self._capability_names) <= 3:
            # Show named capabilities for small sets
            caps_str = ", ".join(f"{name}={val:.2f}" 
                               for name, val in self._capabilities_dict.items())
        else:
            # Show array representation for large sets
            caps_str = np.array2string(self._capabilities_array, precision=2, floatmode="fixed")
        
        return (
            f"Agent(id='{self.agent_id}', load={self.load:.2f}, "
            f"capabilities=({caps_str}), role='{self.role}')"
        )


class AgentPool:
    """
    Enhanced container to manage the swarm of all active micro-cell agents.
    Supports both dictionary and array-based capability systems.
    """

    def __init__(self):
        self.agents: List[Agent] = []
        self._agent_map: Dict[str, Agent] = {}

    def add_agent(self, agent: Agent):
        if agent.agent_id in self._agent_map:
            raise ValueError(f"Agent with ID '{agent.agent_id}' already exists.")
        self.agents.append(agent)
        self._agent_map[agent.agent_id] = agent
        logger.info(f"AgentPool: Added agent '{agent.agent_id}'.")

    def create_and_add_agent(
        self, 
        agent_id: str, 
        capabilities: Union[Dict[str, float], np.ndarray, List[float]],
        load: float = 0.0,
        role: str = "worker"
    ) -> Agent:
        """Convenience method to create and add an agent."""
        agent = Agent(
            agent_id=agent_id,
            capabilities=capabilities,
            _load=load,
            role=role
        )
        self.add_agent(agent)
        return agent

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
    
    def filter_by_capability(self, capability_name: str, min_value: float) -> List[Agent]:
        """Return agents with specific capability above minimum value."""
        return [
            a for a in self.agents 
            if a.has_capability(capability_name) and a.get_capability(capability_name) >= min_value
        ]
    
    def filter_by_role(self, role: str) -> List[Agent]:
        """Return agents with specific role."""
        return [a for a in self.agents if a.role == role]

    @property
    def count(self) -> int:
        return len(self.agents)

    def as_dataframe(self):
        """Export agent data as pandas DataFrame."""
        try:
            import pandas as pd

            data = []
            for agent in self.agents:
                row = {
                    "agent_id": agent.agent_id,
                    "load": agent.load,
                    "role": agent.role,
                    "avg_performance": agent.get_average_performance()
                }
                # Add capability columns
                row.update(agent.capabilities_dict)
                data.append(row)
            
            return pd.DataFrame(data)
        except ImportError:
            logger.warning("Pandas not installed. DataFrame export unavailable.")
            return None

    def __repr__(self) -> str:
        role_counts = {}
        for agent in self.agents:
            role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
        
        role_str = ", ".join(f"{role}: {count}" for role, count in role_counts.items())
        return f"AgentPool(count={self.count}, roles=({role_str}))"

    def find_best_agent(
        self, 
        required_capabilities: Union[Dict[str, float], np.ndarray],
        capability_names: Optional[List[str]] = None
    ) -> Optional[Agent]:
        """
        Enhanced method to find the best agent for given capabilities.
        Supports both dictionary and array formats.
        """
        if len(self.agents) == 0:
            return None
        
        # Convert input to numpy array for computation
        if isinstance(required_capabilities, dict):
            # Dictionary input - match against named capabilities
            req_dict = required_capabilities
            
            # Find agents that have all required capabilities
            valid_agents = []
            for agent in self.agents:
                if all(agent.has_capability(name) for name in req_dict.keys()):
                    valid_agents.append(agent)
            
            if not valid_agents:
                logger.warning("No agents found with all required capabilities")
                return None
            
            # Calculate scores based on capability matching
            best_agent = None
            best_score = float("-inf")
            
            for agent in valid_agents:
                score = 0.0
                total_weight = 0.0
                
                for cap_name, required_level in req_dict.items():
                    agent_level = agent.get_capability(cap_name)
                    # Score based on how well agent meets requirement
                    if required_level > 0:
                        capability_score = min(agent_level / required_level, 1.0)
                    else:
                        capability_score = agent_level
                    
                    score += capability_score * required_level
                    total_weight += required_level
                
                if total_weight > 0:
                    final_score = score / total_weight
                    # Apply load penalty
                    final_score *= (1 - agent.load)
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_agent = agent
            
            return best_agent
            
        else:
            # Array input - use original cosine similarity approach
            required_capabilities = np.asarray(required_capabilities, dtype=float)
            req_norm = np.linalg.norm(required_capabilities)
            
            if req_norm == 0:
                raise ValueError("required_capabilities vector cannot be all zeros.")
            
            best_agent = None
            best_score = float("-inf")
            
            for agent in self.agents:
                # Use cosine similarity
                agent_caps = agent.capabilities_array
                if len(agent_caps) != len(required_capabilities):
                    continue  # Skip agents with incompatible capability dimensions
                
                score = np.dot(agent_caps, required_capabilities) / (
                    np.linalg.norm(agent_caps) * req_norm
                )
                
                # Apply load penalty
                score *= (1 - agent.load)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            return best_agent

    def get_capability_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about capabilities across all agents."""
        if not self.agents:
            return {}
        
        # Get all unique capability names
        all_capabilities = set()
        for agent in self.agents:
            all_capabilities.update(agent.capability_names)
        
        stats = {}
        for cap_name in all_capabilities:
            values = [
                agent.get_capability(cap_name) 
                for agent in self.agents 
                if agent.has_capability(cap_name)
            ]
            
            if values:
                stats[cap_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return stats


# Convenience functions for creating agents
def create_agent_from_dict(agent_id: str, capabilities: Dict[str, float], **kwargs) -> Agent:
    """Create an agent from dictionary capabilities."""
    return Agent(agent_id=agent_id, capabilities=capabilities, **kwargs)

def create_agent_from_array(agent_id: str, capabilities: np.ndarray, **kwargs) -> Agent:
    """Create an agent from array capabilities."""
    return Agent(agent_id=agent_id, capabilities=capabilities, **kwargs)

def create_specialized_agent(
    agent_id: str, 
    specialization: str, 
    level: float = 0.9, 
    other_capabilities: Optional[Dict[str, float]] = None
) -> Agent:
    """Create an agent specialized in one area."""
    capabilities = other_capabilities.copy() if other_capabilities else {}
    capabilities[specialization] = level
    
    # Add some baseline capabilities if not specified
    baseline_caps = {
        "general": 0.5, 
        "communication": 0.6, 
        "reasoning": 0.5
    }
    
    for cap, val in baseline_caps.items():
        if cap not in capabilities:
            capabilities[cap] = val
    
    return Agent(agent_id=agent_id, capabilities=capabilities, role="specialist")

def create_generalist_agent(
    agent_id: str,
    capability_names: List[str],
    base_level: float = 0.7
) -> Agent:
    """Create a generalist agent with uniform capabilities."""
    capabilities = {name: base_level for name in capability_names}
    return Agent(agent_id=agent_id, capabilities=capabilities, role="generalist")


# Backward compatibility
def create_agent_legacy(agent_id: str, capabilities: np.ndarray) -> Agent:
    """Legacy function for backward compatibility."""
    return Agent(agent_id=agent_id, capabilities=capabilities)