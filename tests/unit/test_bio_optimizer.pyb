import numpy as np
import pytest
from src.coordination.bio_optimizer import BioOptimizer

# --- Test the main orchestrator method ---

def test_run_optimization_cycle_returns_correct_structure():
    """
    Tests the main public method to ensure it returns the expected data structure.
    """
    optimizer = BioOptimizer(seed=42)
    # Define a mock system state to pass to the cycle
    system_state = {
        "context": "default",
        "agent_fitness": {"Agent A": 0.8, "Agent B": 0.6},
        "successful_paths": {"(task, Agent A)": 0.9}
    }
    
    result = optimizer.run_optimization_cycle(system_state)

    # Check the overall structure of the output
    assert isinstance(result, dict)
    assert "pso_global_best" in result
    assert "pheromone_levels" in result
    assert "conflict_weights" in result

    # Check the types and shapes of the returned data
    assert isinstance(result["pso_global_best"], np.ndarray)
    assert result["pso_global_best"].shape == (128,)
    assert isinstance(result["pheromone_levels"], dict)
    assert isinstance(result["conflict_weights"], tuple)
    assert len(result["conflict_weights"]) == 2


# --- Test the internal helper methods ---

def test_abc_strategy_logic():
    """
    Tests all three branches of the ABC strategic decision logic.
    """
    optimizer = BioOptimizer()

    # 1. High conflict should favor ACO (0.2, 0.8)
    weights_high_conflict = optimizer._run_abc_strategy(pso_proposal=0.8, aco_proposal=0.2, context="default")
    assert weights_high_conflict == (0.2, 0.8)

    # 2. Low conflict should be balanced (0.5, 0.5)
    weights_low_conflict = optimizer._run_abc_strategy(pso_proposal=0.8, aco_proposal=0.78, context="default")
    assert weights_low_conflict == (0.5, 0.5)

    # 3. "multilingual" context should override conflict and favor PSO (0.75, 0.25)
    weights_context_override = optimizer._run_abc_strategy(pso_proposal=0.8, aco_proposal=0.2, context="multilingual")
    assert weights_context_override == (0.75, 0.25)


def test_pso_tactics_logic():
    """
    Tests the internal PSO tactical method.
    """
    optimizer = BioOptimizer()
    fitness_scores = {"agent1": 0.8, "agent2": 0.95} # agent2 is clearly better
    result = optimizer._run_pso_tactics(fitness_scores)

    assert "pso_global_best" in result
    assert "best_agent" in result
    assert result["best_agent"] == "agent2"
    assert isinstance(result["pso_global_best"], np.ndarray)


def test_aco_memory_update_logic():
    """
    Tests the internal ACO memory update method.
    """
    optimizer = BioOptimizer()
    # Pheromone for path to a1 should be higher due to better success metric
    successful_paths = {"(t1, a1)": 0.9, "(t1, a2)": 0.3}
    pheromones = optimizer._run_aco_memory_update(successful_paths)

    assert isinstance(pheromones, dict)
    assert "(t1, a1)" in pheromones
    assert "(t1, a2)" in pheromones
    assert pheromones["(t1, a1)"] > pheromones["(t1, a2)"]