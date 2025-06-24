import numpy as np
from src.coordination.bio_optimizer import BioOptimizer

def test_pso_returns_global_best():
    optimizer = BioOptimizer()
    fitness_scores = {"agent1": 0.8, "agent2": 0.6}
    result = optimizer.run_pso(fitness_scores)
    assert "pso_global_best" in result
    assert isinstance(result["pso_global_best"], np.ndarray)
    assert result["pso_global_best"].shape == (128,)

def test_aco_returns_pheromone_levels():
    optimizer = BioOptimizer()
    successful_paths = {"(t1, a1)": 0.9}
    result = optimizer.run_aco(successful_paths)
    assert "pheromone_levels" in result
    levels = result["pheromone_levels"]
    assert isinstance(levels, dict)
    for k, v in levels.items():
        assert isinstance(k, str)
        assert 0.0 <= v <= 1.0

def test_abc_high_conflict():
    optimizer = BioOptimizer()
    conflict_score = 0.7
    w_pso, w_aco = optimizer.run_abc(conflict_score)
    # When high conflict, ACO weight should be higher
    assert w_aco > w_pso
    assert 0 <= w_pso <= 1
    assert 0 <= w_aco <= 1

def test_abc_low_conflict():
    optimizer = BioOptimizer()
    conflict_score = 0.2
    w_pso, w_aco = optimizer.run_abc(conflict_score)
    # Should be roughly balanced
    assert abs(w_pso - w_aco) < 0.01
    assert 0 <= w_pso <= 1
    assert 0 <= w_aco <= 1

