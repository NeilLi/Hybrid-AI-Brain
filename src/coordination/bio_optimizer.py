#!/usr/bin/env python3
"""
src/coordination/bio_optimizer.py

Implements a hierarchical, bio-inspired optimization strategy (ABC -> PSO -> ACO) 
for heuristic input to the GNN coordinator, as per the "Hybrid AI Brain" paper.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional

# --- Setup Logging ---
logger = logging.getLogger("hybrid_ai_brain.bio_optimizer")
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class BioOptimizer:
    """
    Implements a hierarchical strategy where ABC governs PSO and ACO.
    """
    def __init__(
        self,
        pso_params: Optional[Dict[str, Any]] = None,
        aco_params: Optional[Dict[str, Any]] = None,
        abc_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None
    ):
        self.pso_params = pso_params or {"dim": 128, "w": 0.7, "c1": 1.5, "c2": 1.5}
        self.aco_params = aco_params or {"evaporation": 0.5}
        self.abc_params = abc_params or {"conflict_threshold": 0.05}
        if seed is not None:
            np.random.seed(seed)
        logger.info("Hierarchical BioOptimizer (ABC -> PSO -> ACO) initialized.")

    def _run_abc_strategy(self, pso_proposal: float, aco_proposal: float, context: str) -> Tuple[float, float]:
        """
        [MACROSCOPIC] ABC as the strategist and meta-optimizer.
        Determines the conflict resolution weights based on proposals and context.
        """
        conflict_score = abs(pso_proposal - aco_proposal)
        logger.info(f"ABC (Strategy): Received proposals (PSO: {pso_proposal:.2f}, ACO: {aco_proposal:.2f}). Conflict score: {conflict_score:.2f}")

        if context == "multilingual":
            logger.info("  - ABC: 'Multilingual' context detected. Prioritizing PSO for generalist capabilities.")
            return (0.75, 0.25)
        
        if conflict_score > self.abc_params["conflict_threshold"]:
            logger.info("  - ABC: High conflict detected. Prioritizing historically successful paths (ACO).")
            return (0.2, 0.8)
        
        logger.info("  - ABC: Low conflict. Using balanced weights.")
        return (0.5, 0.5)

    def _run_pso_tactics(self, agent_fitness_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        [MESOSCOPIC] PSO as the tactician for a sub-swarm.
        Returns the best-found parameters ('g_best') for the current task.
        """
        logger.info("PSO (Tactics): Optimizing team strategy...")
        dim = self.pso_params.get("dim", 128)
        if not agent_fitness_scores:
            logger.warning("  - PSO: No fitness scores provided. Returning random vector.")
            return {"pso_global_best": np.random.rand(dim), "best_agent": None}

        best_agent = max(agent_fitness_scores, key=agent_fitness_scores.get)
        g_best = np.ones(dim) * (hash(best_agent) % 100 / 100.0)
        logger.info(f"  - PSO: Tactical solution found, favoring '{best_agent}'.")
        return {"pso_global_best": g_best, "best_agent": best_agent}

    def _run_aco_memory_update(self, successful_paths: Dict[str, float]) -> Dict[str, float]:
        """
        [MICROSCOPIC] ACO as the foundational memory layer.
        Updates the pheromone map based on successful outcomes.
        """
        logger.info("ACO (Memory): Updating persistent pheromone map...")
        evaporation = self.aco_params.get("evaporation", 0.5)
        pheromone_levels = {}
        for path, success_metric in successful_paths.items():
            deposit = (1 - evaporation) * success_metric
            # Assume a baseline previous pheromone level for the demo
            new_pheromone = (evaporation * 0.1) + deposit
            pheromone_levels[path] = new_pheromone
            logger.debug(f"  - ACO: Path '{path}' reinforced with new pheromone level {new_pheromone:.3f}.")
        return pheromone_levels

    def run_optimization_cycle(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates one full hierarchical optimization cycle.
        """
        logger.info("--- Starting New Bio-Optimization Cycle ---")
        
        # 1. PSO runs its tactical optimization based on current agent performance
        agent_fitness = system_state.get("agent_fitness", {})
        pso_result = self._run_pso_tactics(agent_fitness)
        pso_proposal_strength = agent_fitness.get(pso_result["best_agent"], 0)

        # 2. ACO provides its proposal based on historical path success
        # For this demo, we'll find the best historical path from the input
        historical_paths = system_state.get("successful_paths", {})
        best_historical_path_strength = max(historical_paths.values()) if historical_paths else 0

        # 3. ABC acts as the high-level strategist to resolve conflict
        context = system_state.get("context", "default")
        weights = self._run_abc_strategy(pso_proposal_strength, best_historical_path_strength, context)

        # 4. ACO updates its memory based on the most recent successes
        pheromones = self._run_aco_memory_update(historical_paths)
        
        logger.info("--- Bio-Optimization Cycle Complete ---")
        
        # The final output to be consumed by the GNN
        return {
            "pso_global_best": pso_result["pso_global_best"],
            "pheromone_levels": pheromones,
            "conflict_weights": weights,
        }

# --- Demo Block ---
if __name__ == "__main__":
    print("-" * 65)
    print("DEMO: Simulating Hierarchical Optimization (ABC -> PSO -> ACO)")
    print("-" * 65)

    optimizer = BioOptimizer(seed=42)

    # Define the initial state corresponding to the paper's "Sentiment Analysis" scenario
    sentiment_analysis_state = {
        "context": "default",
        "agent_fitness": {"Agent A": 0.78, "Agent B": 0.65},      # PSO will propose Agent A
        "successful_paths": {"(task, Agent A)": 0.4, "(task, Agent B)": 0.85} # ACO's history favors Agent B
    }

    print("\n[CYCLE 1: 'Sentiment Analysis' Scenario]\n")
    cycle_1_output = optimizer.run_optimization_cycle(sentiment_analysis_state)
    print("\n==> Cycle 1 Output to GNN:", cycle_1_output)

    # Define the next state corresponding to the "Multilingual Analysis" context shift
    multilingual_state = {
        "context": "multilingual", # <-- The key change
        "agent_fitness": {"Agent A": 0.9, "Agent B": 0.5},
        "successful_paths": {"(task, Agent A)": 0.9} # New successful paths reinforce Agent A
    }

    print("\n[CYCLE 2: 'Multilingual Analysis' Context Shift]\n")
    cycle_2_output = optimizer.run_optimization_cycle(multilingual_state)
    print("\n==> Cycle 2 Output to GNN:", cycle_2_output)
    
    print("\n" + "-" * 65)