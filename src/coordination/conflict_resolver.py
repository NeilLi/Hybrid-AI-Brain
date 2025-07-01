#!/usr/bin/env python3
"""
src/coordination/conflict_resolver.py

Implements the conflict resolution protocol that merges signals from
different bio-inspired algorithms, as per the Hybrid AI Brain paper.
"""

import logging
from typing import Dict, Tuple

# --- Setup Logging ---
logger = logging.getLogger("hybrid_ai_brain.conflict_resolver")
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class ConflictResolver:
    """
    Resolves conflicting edge weights from PSO and ACO using ABC-provided weights,
    implementing Equation 9 of the Hybrid AI Brain paper.
    """
    def __init__(self):
        logger.info("ConflictResolver initialized.")

    def resolve(
        self,
        pso_weights: Dict[str, float],
        aco_weights: Dict[str, float],
        mixing_weights: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Resolves final edge weights:
        w_ij = λ_PSO * w_ij_PSO + λ_ACO * w_ij_ACO

        Args:
            pso_weights: Proposed edge weights from PSO.
            aco_weights: Proposed edge weights from ACO.
            mixing_weights: Tuple (λ_PSO, λ_ACO) from ABC. Should sum to 1.

        Returns:
            Dictionary of resolved edge weights.
        """
        lambda_pso, lambda_aco = mixing_weights
        total_lambda = lambda_pso + lambda_aco
        if abs(total_lambda - 1.0) > 1e-6:
            # Normalize weights if they don't sum to 1, as a safeguard.
            logger.warning(
                f"Mixing weights ({lambda_pso}, {lambda_aco}) do not sum to 1. Normalizing."
            )
            lambda_pso /= total_lambda
            lambda_aco /= total_lambda

        logger.info(f"Resolving with weights λ_PSO={lambda_pso:.2f}, λ_ACO={lambda_aco:.2f}.")

        # Get the union of all edges proposed by either algorithm
        all_edges = set(pso_weights.keys()) | set(aco_weights.keys())

        final_weights = {}
        for edge in all_edges:
            w_pso = pso_weights.get(edge, 0.0) # Default to 0 if an edge is not in the proposal
            w_aco = aco_weights.get(edge, 0.0)
            resolved_weight = (lambda_pso * w_pso) + (lambda_aco * w_aco)
            final_weights[edge] = resolved_weight
            logger.debug(f"  - Edge '{edge}': Resolved weight = {resolved_weight:.3f}")

        logger.info("Unified edge weights created.")
        return final_weights

def main():
    """
    Demo function updated to simulate the "Detailed Conflict Resolution Scenario"
    from Section 6.4.1 of the paper.
    """
    logger.info("====== Coordination Layer: ConflictResolver Demo ======")
    resolver = ConflictResolver()

    # --- Setup the scenario from the paper ---
    # PSO tactically favors the generalist Agent A for coordination efficiency.
    pso_proposals = {
        "(sentiment, Agent A)": 0.78, 
        "(sentiment, Agent B)": 0.65 
    }
    
    # ACO's memory/history strongly favors the specialist Agent B.
    aco_proposals = {
        "(sentiment, Agent A)": 0.40, 
        "(sentiment, Agent B)": 0.85
    }
    
    print("\n--- [SCENARIO 1: High-Conflict 'Sentiment Analysis'] ---")
    # In a high-conflict scenario, ABC's strategy is to favor history (ACO).
    abc_weights_conflict = (0.2, 0.8)
    final_weights_1 = resolver.resolve(pso_proposals, aco_proposals, abc_weights_conflict)
    print("Final resolved weights (favoring ACO):")
    for edge, weight in final_weights_1.items():
        print(f"  {edge}: {weight:.3f}")
    print(f"==> Result: The system prioritizes '{max(final_weights_1, key=final_weights_1.get)}'")


    print("\n--- [SCENARIO 2: Context Shift to 'Multilingual Analysis'] ---")
    # When the context shifts, ABC's strategy changes to favor the generalist (PSO).
    abc_weights_multilingual = (0.75, 0.25)
    final_weights_2 = resolver.resolve(pso_proposals, aco_proposals, abc_weights_multilingual)
    print("Final resolved weights (favoring PSO):")
    for edge, weight in final_weights_2.items():
        print(f"  {edge}: {weight:.3f}")
    print(f"==> Result: The system now prioritizes '{max(final_weights_2, key=final_weights_2.get)}'")
    
    logger.info("\n====== Demo complete. ======")


if __name__ == "__main__":
    main()