#!/usr/bin/env python3
"""
src/safety/graph_mask.py

Implements the GraphMask safety layer, responsible for filtering unsafe edges
in the task execution graph based on learned models and statistical validation.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional

# --- Constants from the paper's safety analysis ---
TAU_SAFE = 0.7       # Default safety threshold (τ_safe)
SAFETY_SAMPLES = 59  # n_samples for Pr(false-block) <= 10^-4

logger = logging.getLogger("hybrid_ai_brain.safety.graph_mask")
logging.basicConfig(level=logging.INFO)

class GraphMask:
    """
    Implements the GraphMask safety mechanism for explainable and robust
    safety filtering of task-agent assignments (edges).
    """
    def __init__(self, mask_model: Optional[Any] = None, rng_seed: Optional[int] = None):
        """
        Args:
            mask_model: Optional pre-trained mask generator model (for real use).
            rng_seed: Optional random seed for reproducible testing.
        """
        self._mask_generator_model: Any = mask_model or self._load_model()
        self.rng = np.random.default_rng(rng_seed)
        logger.info("GraphMask safety system initialized.")

    def _load_model(self) -> Any:
        logger.info("Loading trained mask generator model (simulated).")
        return "trained_model_placeholder"

    def is_edge_safe(
        self,
        edge_features: Dict[str, Any],
        safety_samples: int = SAFETY_SAMPLES,
        tau_safe: float = TAU_SAFE
    ) -> bool:
        """
        Evaluates edge safety using a simulated mask generator and statistical validation.

        Returns:
            True if the edge is considered safe, False if blocked.
        """
        logger.info(f"Evaluating edge with n={safety_samples} samples and τ_safe={tau_safe}.")

        # Simulate empirical unsafe scores from the model (in real use, query the actual model)
        empirical_unsafe_scores = self.rng.uniform(0.0, 1.0, size=safety_samples)
        p_hat = float(np.mean(empirical_unsafe_scores))

        if p_hat > tau_safe:
            logger.warning(
                f"UNSAFE: Edge blocked. (Empirical Mean Pr(unsafe)={p_hat:.4f} > {tau_safe})"
            )
            return False
        else:
            logger.info(
                f"SAFE: Edge approved. (Empirical Mean Pr(unsafe)={p_hat:.4f} <= {tau_safe})"
            )
            return True

def main():
    logger.info("====== Safety Layer: GraphMask Demo ======")
    # Set a seed for reproducibility in demo/test
    graph_mask = GraphMask(rng_seed=42)

    # Simulate evaluating a clearly safe edge
    logger.info("--- Evaluating a likely SAFE edge ---")
    safe_edge_features = {"risk_score": 0.1, "data_type": "public"}
    is_safe = graph_mask.is_edge_safe(safe_edge_features)
    print(f"Safe edge result: {'SAFE' if is_safe else 'BLOCKED'}")

    # Simulate evaluating a potentially unsafe edge
    logger.info("--- Evaluating a likely UNSAFE edge ---")
    unsafe_edge_features = {"risk_score": 0.9, "data_type": "PII"}
    is_safe = graph_mask.is_edge_safe(unsafe_edge_features)
    print(f"Unsafe edge result: {'SAFE' if is_safe else 'BLOCKED'}")

    # Evaluate with PRECISION domain parameters (stricter)
    logger.info("--- Evaluating with PRECISION domain parameters ---")
    is_safe = graph_mask.is_edge_safe(
        edge_features=unsafe_edge_features,
        safety_samples=116,
        tau_safe=0.8
    )
    print(f"Precision domain edge result: {'SAFE' if is_safe else 'BLOCKED'}")

    logger.info("✅ graph_mask.py executed successfully!")

if __name__ == "__main__":
    main()
