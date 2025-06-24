#!/usr/bin/env python3
"""
src/coordination/conflict_resolver.py

Implements the conflict resolution protocol that merges signals from
different bio-inspired algorithms.
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger("hybrid_ai_brain.conflict_resolver")
logging.basicConfig(level=logging.INFO)

class ConflictResolver:
    """
    Resolves conflicting edge weights from PSO and ACO using ABC-provided weights,
    implementing Equation 39 of the Hybrid AI Brain paper.
    """
    def __init__(self):
        logger.info("ConflictResolver initialized.")

    def resolve(
        self,
        pso_weights: Dict[str, float],
        aco_weights: Dict[str, float],
        mixing_weights: Tuple[float, float],
        normalize: bool = False
    ) -> Dict[str, float]:
        """
        Resolves final edge weights:
        w_ij = λ_PSO * w_ij_PSO + λ_ACO * w_ij_ACO

        Args:
            pso_weights: Proposed edge weights from PSO.
            aco_weights: Proposed edge weights from ACO.
            mixing_weights: Tuple (λ_PSO, λ_ACO) from ABC. Should sum to 1.
            normalize: If True, normalize output weights to [0,1] range.

        Returns:
            Dictionary of resolved edge weights.
        """
        lambda_pso, lambda_aco = mixing_weights
        total_lambda = lambda_pso + lambda_aco
        if abs(total_lambda - 1.0) > 1e-6:
            logger.warning("Mixing weights do not sum to 1. Normalizing.")
            lambda_pso /= total_lambda
            lambda_aco /= total_lambda

        logger.info(f"Resolving with weights λ_PSO={lambda_pso:.2f}, λ_ACO={lambda_aco:.2f}.")

        # Union of all edges
        all_edges = set(pso_weights.keys()) | set(aco_weights.keys())

        final_weights = {}
        for edge in all_edges:
            w_pso = pso_weights.get(edge, 0.0)
            w_aco = aco_weights.get(edge, 0.0)
            resolved = (lambda_pso * w_pso) + (lambda_aco * w_aco)
            final_weights[edge] = resolved

        if normalize and final_weights:
            max_w = max(final_weights.values())
            min_w = min(final_weights.values())
            range_w = max_w - min_w if max_w != min_w else 1
            for k in final_weights:
                final_weights[k] = (final_weights[k] - min_w) / range_w

        logger.info("Unified edge weights created.")
        return final_weights

def main():
    logger.info("====== Coordination Layer: ConflictResolver Demo ======")
    resolver = ConflictResolver()

    pso_props = {"(t1,a1)": 0.9, "(t1,a2)": 0.3}
    aco_props = {"(t1,a1)": 0.4, "(t1,a2)": 0.8}

    # Example 1: Balanced weights
    abc_weights = (0.5, 0.5)
    logger.info("--- Resolving with balanced weights ---")
    final = resolver.resolve(pso_props, aco_props, abc_weights)
    print("Final resolved weights:", {k: f"{v:.2f}" for k, v in final.items()})

    # Example 2: Favor ACO (after detected conflict)
    abc_weights_adjusted = (0.2, 0.8)
    logger.info("--- Resolving with ABC favoring ACO ---")
    final_adjusted = resolver.resolve(pso_props, aco_props, abc_weights_adjusted)
    print("Final resolved weights:", {k: f"{v:.2f}" for k, v in final_adjusted.items()})

    # Example 3: Normalization
    logger.info("--- Resolving with normalization ---")
    final_norm = resolver.resolve(pso_props, aco_props, abc_weights, normalize=True)
    print("Final resolved weights (normalized):", {k: f"{v:.2f}" for k, v in final_norm.items()})

if __name__ == "__main__":
    main()
