#!/usr/bin/env python3
"""
src/coordination/bio_optimizer.py

Implements bio-inspired optimization (PSO, ACO, ABC) for heuristic input to GNN coordinator.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger("hybrid_ai_brain.bio_optimizer")
logging.basicConfig(level=logging.INFO)

class BioOptimizer:
    """
    Implements PSO, ACO, and ABC metaheuristics for agent-task assignment optimization.
    """
    def __init__(
        self,
        pso_params: Optional[Dict[str, Any]] = None,
        aco_params: Optional[Dict[str, Any]] = None,
        abc_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        self.pso_params = pso_params or {"swarm_size": 30, "dim": 128, "w": 0.7, "c1": 1.5, "c2": 1.5}
        self.aco_params = aco_params or {"evaporation": 0.5, "alpha": 1.0, "beta": 2.0}
        self.abc_params = abc_params or {"population": 20, "limit": 10}
        if seed is not None:
            np.random.seed(seed)
        logger.info("BioOptimizer (PSO, ACO, ABC) initialized.")

    def run_pso(self, current_fitness_scores: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Runs a PSO step to find optimal agent parameters.
        Returns: dict with the 'pso_global_best' vector.
        """
        logger.info("BioOptimizer: Running PSO step...")
        dim = self.pso_params.get("dim", 128)
        # Simulate global best as argmax of provided fitness
        if current_fitness_scores:
            best_key = max(current_fitness_scores, key=current_fitness_scores.get)
            # For demo, encode best_key as a simple embedding
            global_best = np.ones(dim) * hash(best_key) % 100 / 100
        else:
            global_best = np.random.rand(dim)
        logger.debug(f"  - PSO: Global best vector computed.")
        return {"pso_global_best": global_best}

    def run_aco(self, successful_paths: Dict[str, float]) -> Dict[str, float]:
        """
        Runs an ACO step, returning updated pheromone levels for paths.
        """
        logger.info("BioOptimizer: Running ACO step (updating pheromones)...")
        evaporation = self.aco_params.get("evaporation", 0.5)
        pheromone_levels = {}
        for path, success in successful_paths.items():
            # Simple pheromone update rule: new = old*evap + success*(1-evap)
            prev_pheromone = np.random.uniform(0.2, 0.8)
            new_pheromone = evaporation * prev_pheromone + (1-evaporation) * success
            pheromone_levels[path] = new_pheromone
        logger.debug(f"  - ACO: Pheromone levels updated.")
        return {"pheromone_levels": pheromone_levels}

    def run_abc(self, conflict_score: float) -> Tuple[float, float]:
        """
        Runs an ABC step to tune conflict resolution weights.
        Returns: (lambda_PSO, lambda_ACO)
        """
        logger.info("BioOptimizer: Running ABC step for meta-optimization...")
        if conflict_score > 0.5:
            logger.info("  - ABC: High conflict. Exploring new weights.")
            # Example: Favor ACO in high-conflict scenario
            return (0.3, 0.7)
        else:
            # Balanced weights for low conflict
            return (0.5, 0.5)

    def __repr__(self):
        return (f"BioOptimizer(PSO={self.pso_params}, ACO={self.aco_params}, "
                f"ABC={self.abc_params})")

# --- Demo Block ---
if __name__ == "__main__":
    optimizer = BioOptimizer(seed=42)
    pso_result = optimizer.run_pso({"agent1": 0.6, "agent2": 0.9})
    print("PSO:", pso_result)
    aco_result = optimizer.run_aco({"(t1,a2)": 1.0, "(t2,a1)": 0.8, "(t1,a1)": 0.1})
    print("ACO:", aco_result)
    abc_result = optimizer.run_abc(0.65)
    print("ABC:", abc_result)
