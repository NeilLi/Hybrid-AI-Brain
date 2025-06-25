#!/usr/bin/env python3
"""
src/safety/graph_mask.py

Implements the GraphMask safety layer, responsible for filtering unsafe edges
in the task execution graph based on learned models and statistical validation.
"""

import numpy as np
from typing import Any, Dict
import logging

# --- Constants from the paper's safety analysis ---
TAU_SAFE = 0.7       # Default safety threshold (τ_safe) [source: 120]
SAFETY_SAMPLES = 59  # n_samples for Pr(false-block) <= 10^-4 [source: 115, 257, 802]

class GraphMask:
    """
    Implements the GraphMask safety mechanism for explainable and robust
    safety filtering of task-agent assignments (edges). [source: 26, 622]
    """
    def __init__(self):
        """
        Initializes the GraphMask system.
        In a real implementation, this would load a trained mask generator model.
        """
        self._mask_generator_model: Any = self._load_model()
        logging.info("GraphMask safety system initialized.")

    def _load_model(self) -> Any:
        """A private method to simulate loading a pre-trained mask generator model."""
        logging.info("GraphMask: Loading trained mask generator model (simulated).")
        return "trained_model_placeholder"

    def is_edge_safe(
        self,
        edge_features: Dict[str, Any],
        safety_samples: int = SAFETY_SAMPLES,
        tau_safe: float = TAU_SAFE
    ) -> bool:
        """
        Evaluates if an edge is safe by implementing the blocking predicate.
        This function performs the core safety check described in Section 3.3.
        
        Args:
            edge_features: The features associated with the graph edge to evaluate.
            safety_samples: The number of Monte Carlo samples (n) to draw.
            tau_safe: The safety threshold for blocking.

        Returns:
            True if the edge is considered safe, False if it should be blocked.
        """
        logging.info(f"GraphMask: Evaluating edge with n={safety_samples} samples and τ_safe={tau_safe}.")

        # --- FIX: Make the simulation sensitive to the input risk ---
        # The SafetyMonitor adds the 'heuristic_risk' to the features.
        # We use this to center the distribution of our simulated model's outputs.
        heuristic_risk = edge_features.get('heuristic_risk', 0.5)

        # Simulate a model's uncertain output using a normal distribution
        # centered around the heuristic risk.
        loc = heuristic_risk  # The mean of the distribution
        scale = 0.15          # The standard deviation (model uncertainty)
        
        empirical_unsafe_scores = np.random.normal(loc=loc, scale=scale, size=safety_samples)
        
        # Clip the scores to ensure they are valid probabilities between 0 and 1.
        empirical_unsafe_scores = np.clip(empirical_unsafe_scores, 0, 1)

        # The empirical mean unsafe probability (p_hat)
        p_hat = np.mean(empirical_unsafe_scores)
        
        # The blocking predicate: block(e) = I[p_hat > τ_safe] [source: 119]
        if p_hat > tau_safe:
            logging.info(f"UNSAFE: Edge blocked. (Empirical Mean Pr(unsafe)={p_hat:.4f} > {tau_safe})")
            return False
        else:
            logging.info(f"SAFE: Edge approved. (Empirical Mean Pr(unsafe)={p_hat:.4f} <= {tau_safe})")
            return True
