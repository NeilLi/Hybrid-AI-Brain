#!/usr/bin/env python3
"""
src/core/match_score.py

Enhanced Match Score function supporting both dictionary and array capabilities.
Implements the core Agent-Task Match Score function from Definition 3.1
of the Hybrid AI Brain paper with flexible capability formats.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List
from .agent_pool import Agent

def match_score(agent, task):
    """Simple compatibility function for array inputs."""
    norm_agent = np.linalg.norm(agent)
    norm_task = np.linalg.norm(task)
    if norm_agent == 0 or norm_task == 0:
        return 0.0
    score = np.dot(agent, task) / (norm_agent * norm_task)
    return float(score)

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid activation."""
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(-x))

def calculate_match_score(
    agent: Agent,
    task_node: Union[Dict[str, Any], np.ndarray],
    path_risk: float = 0.0,
    beta: float = 2.0,
    theta: float = 0.5,
    alpha: float = 1.5,
    lambda_risk: float = 1.0,
    debug: bool = False,
) -> float:
    """
    Enhanced match score calculation supporting both dictionary and array formats.

    Args:
        agent: The agent being evaluated.
        task_node: Task requirements - can be dict with 'required_capabilities' or numpy array.
        path_risk: The accumulated risk score of the execution path to this task.
        beta, theta, alpha, lambda_risk: Model constants.
        debug: If True, prints internal calculation steps.

    Returns:
        A float score between 0 and 1 representing the match quality.
    """
    if not isinstance(agent, Agent):
        raise TypeError("agent must be an instance of the Agent class.")
    
    # Extract required capabilities from task_node
    if isinstance(task_node, dict):
        if "required_capabilities" in task_node:
            required_caps = task_node["required_capabilities"]
        else:
            # Assume the dict itself contains the requirements
            required_caps = task_node
    elif isinstance(task_node, (list, tuple, np.ndarray)):
        # Direct array input
        required_caps = task_node
    else:
        raise ValueError(
            "task_node must be dict with 'required_capabilities', "
            "dict of requirements, or array-like object."
        )
    
    # Calculate match score based on requirement format
    if isinstance(required_caps, dict):
        # Dictionary-based matching
        return _calculate_dict_match_score(
            agent, required_caps, path_risk, beta, theta, alpha, lambda_risk, debug
        )
    else:
        # Array-based matching
        return _calculate_array_match_score(
            agent, required_caps, path_risk, beta, theta, alpha, lambda_risk, debug
        )

def _calculate_dict_match_score(
    agent: Agent,
    required_capabilities: Dict[str, float],
    path_risk: float,
    beta: float,
    theta: float,
    alpha: float,
    lambda_risk: float,
    debug: bool
) -> float:
    """Calculate match score using dictionary-based capability matching."""
    
    # Check if agent has all required capabilities
    missing_capabilities = []
    for cap_name in required_capabilities:
        if not agent.has_capability(cap_name):
            missing_capabilities.append(cap_name)
    
    if missing_capabilities:
        if debug:
            print(f"[dict_match_score] agent_id={agent.agent_id}")
            print(f"  Missing capabilities: {missing_capabilities}")
            print(f"  final_score = 0.0000")
        return 0.0
    
    # Calculate weighted capability match
    total_score = 0.0
    total_weight = 0.0
    
    for cap_name, required_level in required_capabilities.items():
        agent_level = agent.get_capability(cap_name)
        weight = required_level  # Use requirement level as weight
        
        if required_level > 0:
            # Calculate how well agent meets this requirement
            capability_ratio = agent_level / required_level
            capability_match = min(capability_ratio, 1.0)  # Cap at 1.0
        else:
            # If requirement is 0, any capability level is perfect
            capability_match = 1.0
        
        total_score += capability_match * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        capability_match = total_score / total_weight
    else:
        capability_match = 0.0
    
    # Apply sigmoid transformation
    capability_match = _sigmoid(beta * (capability_match - theta))
    
    # Apply load and risk penalties
    load_penalty = (1 - agent.load) ** alpha
    risk_penalty = np.exp(-lambda_risk * path_risk)
    final_score = capability_match * load_penalty * risk_penalty
    
    if debug:
        print(f"[dict_match_score] agent_id={agent.agent_id}")
        print(f"  capability_match = {capability_match:.4f}")
        print(f"  load_penalty = {load_penalty:.4f}")
        print(f"  risk_penalty = {risk_penalty:.4f}")
        print(f"  final_score = {final_score:.4f}")
    
    return float(final_score)

def _calculate_array_match_score(
    agent: Agent,
    required_capabilities: Union[List, np.ndarray],
    path_risk: float,
    beta: float,
    theta: float,
    alpha: float,
    lambda_risk: float,
    debug: bool
) -> float:
    """Calculate match score using array-based capability matching."""
    
    # Convert to numpy arrays
    r_t = np.asarray(required_capabilities, dtype=float)
    c_i = agent.capabilities_array
    l_i = float(np.clip(agent.load, 0.0, 1.0))

    if r_t.shape != c_i.shape:
        raise ValueError(
            f"Shape mismatch: task required_capabilities {r_t.shape} vs. "
            f"agent capabilities {c_i.shape}"
        )

    # Compute score using original method
    dot_product = float(np.dot(r_t, c_i))
    capability_match = _sigmoid(beta * (dot_product - theta))
    load_penalty = (1 - l_i) ** alpha
    risk_penalty = np.exp(-lambda_risk * path_risk)
    final_score = capability_match * load_penalty * risk_penalty

    if debug:
        print(f"[array_match_score] agent_id={agent.agent_id}")
        print(f"  dot_product = {dot_product:.4f}")
        print(f"  capability_match = {capability_match:.4f}")
        print(f"  load_penalty = {load_penalty:.4f}")
        print(f"  risk_penalty = {risk_penalty:.4f}")
        print(f"  final_score = {final_score:.4f}")

    return float(final_score)

def batch_match_scores(
    agents: List[Agent],
    task_node: Union[Dict[str, Any], np.ndarray],
    path_risk: float = 0.0,
    beta: float = 2.0,
    theta: float = 0.5,
    alpha: float = 1.5,
    lambda_risk: float = 1.0,
) -> np.ndarray:
    """Compute match scores for a list of agents against a single task."""
    scores = []
    for agent in agents:
        scores.append(
            calculate_match_score(
                agent, task_node, path_risk, beta, theta, alpha, lambda_risk
            )
        )
    return np.array(scores)

def find_best_matches(
    agents: List[Agent],
    task_node: Union[Dict[str, Any], np.ndarray],
    top_k: int = 3,
    min_score: float = 0.1,
    **match_params
) -> List[tuple]:
    """
    Find the best matching agents for a task.
    
    Returns:
        List of (agent, score) tuples sorted by score (highest first)
    """
    scores = batch_match_scores(agents, task_node, **match_params)
    
    # Create list of (agent, score) pairs
    agent_scores = list(zip(agents, scores))
    
    # Filter by minimum score
    agent_scores = [(agent, score) for agent, score in agent_scores if score >= min_score]
    
    # Sort by score (descending)
    agent_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return agent_scores[:top_k]

def analyze_capability_gaps(
    agent: Agent,
    task_requirements: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze where an agent falls short of task requirements.
    
    Returns:
        Dictionary with gap analysis
    """
    gaps = {}
    total_gap = 0.0
    
    for cap_name, required_level in task_requirements.items():
        if agent.has_capability(cap_name):
            agent_level = agent.get_capability(cap_name)
            gap = max(0, required_level - agent_level)
            coverage = min(agent_level / required_level, 1.0) if required_level > 0 else 1.0
        else:
            gap = required_level
            coverage = 0.0
        
        gaps[cap_name] = {
            "required": required_level,
            "agent_has": agent.get_capability(cap_name) if agent.has_capability(cap_name) else 0.0,
            "gap": gap,
            "coverage": coverage
        }
        total_gap += gap
    
    # Overall analysis
    total_required = sum(task_requirements.values())
    overall_coverage = 1.0 - (total_gap / total_required) if total_required > 0 else 1.0
    
    return {
        "capability_gaps": gaps,
        "total_gap": total_gap,
        "overall_coverage": overall_coverage,
        "missing_capabilities": [
            cap for cap in task_requirements 
            if not agent.has_capability(cap)
        ]
    }

# Convenience functions for different matching strategies
def simple_cosine_match(agent: Agent, task_requirements: np.ndarray) -> float:
    """Simple cosine similarity matching."""
    agent_caps = agent.capabilities_array
    if len(agent_caps) != len(task_requirements):
        return 0.0
    
    norm_agent = np.linalg.norm(agent_caps)
    norm_task = np.linalg.norm(task_requirements)
    
    if norm_agent == 0 or norm_task == 0:
        return 0.0
    
    return float(np.dot(agent_caps, task_requirements) / (norm_agent * norm_task))

def weighted_capability_match(agent: Agent, task_requirements: Dict[str, float]) -> float:
    """Weighted capability matching without sigmoid transformation."""
    if not all(agent.has_capability(cap) for cap in task_requirements):
        return 0.0
    
    total_score = 0.0
    total_weight = 0.0
    
    for cap_name, required_level in task_requirements.items():
        agent_level = agent.get_capability(cap_name)
        weight = required_level
        
        if required_level > 0:
            score = min(agent_level / required_level, 1.0)
        else:
            score = 1.0
        
        total_score += score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0.0