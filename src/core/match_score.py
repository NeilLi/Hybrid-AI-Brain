#!/usr/bin/env python3
"""
src/core/match_score.py

Implements the core Agent-Task Match Score function from Definition 3.1
of the Hybrid AI Brain paper.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from .agent_pool import Agent

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid activation."""
    # For large negative x, exp(-x) can overflow, so use np.clip
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(-x))

def calculate_match_score(
    agent: Agent,
    task_node: Dict[str, Any],
    path_risk: float = 0.0,
    beta: float = 2.0,
    theta: float = 0.5,
    alpha: float = 1.5,
    lambda_risk: float = 1.0,
    debug: bool = False,
) -> float:
    """
    Calculates the unified agent-task compatibility score based on Definition 3.1.

    Args:
        agent: The agent being evaluated.
        task_node: The task node data, must contain 'required_capabilities'.
        path_risk: The accumulated risk score of the execution path to this task.
        beta, theta, alpha, lambda_risk: Model constants (overrides config).
        debug: If True, prints internal calculation steps.

    Returns:
        A float score between 0 and 1 representing the match quality.
    """
    if not isinstance(agent, Agent):
        raise TypeError("agent must be an instance of the Agent class.")
    if "required_capabilities" not in task_node:
        raise ValueError("task_node must contain 'required_capabilities'.")

    # Extract/validate
    r_t = np.asarray(task_node["required_capabilities"], dtype=float)
    c_i = np.asarray(agent.capabilities, dtype=float)
    l_i = float(np.clip(agent.load, 0.0, 1.0))

    if r_t.shape != c_i.shape:
        raise ValueError(
            f"Shape mismatch: task required_capabilities {r_t.shape} vs. agent capabilities {c_i.shape}"
        )

    # Compute score
    dot_product = float(np.dot(r_t, c_i))
    capability_match = _sigmoid(beta * (dot_product - theta))
    load_penalty = (1 - l_i) ** alpha
    risk_penalty = np.exp(-lambda_risk * path_risk)
    final_score = capability_match * load_penalty * risk_penalty

    if debug:
        print(f"[match_score] agent_id={agent.id}")
        print(f"  dot_product = {dot_product:.4f}")
        print(f"  capability_match = {capability_match:.4f}")
        print(f"  load_penalty = {load_penalty:.4f}")
        print(f"  risk_penalty = {risk_penalty:.4f}")
        print(f"  final_score = {final_score:.4f}")

    return float(final_score)

# --- Optional: Batch version for vectorized calculation (advanced) ---
def batch_match_scores(
    agents: list[Agent],
    task_node: Dict[str, Any],
    path_risk: float = 0.0,
    beta: float = 2.0,
    theta: float = 0.5,
    alpha: float = 1.5,
    lambda_risk: float = 1.0,
) -> np.ndarray:
    """Compute match scores for a list of agents against a single task."""
    r_t = np.asarray(task_node["required_capabilities"], dtype=float)
    scores = []
    for agent in agents:
        scores.append(
            calculate_match_score(
                agent, task_node, path_risk, beta, theta, alpha, lambda_risk
            )
        )
    return np.array(scores)
