#!/usr/bin/env python3
"""
tests/unit/test_gnn_coordinator.py

Unit tests for the GNNCoordinator, updated to be fully deterministic by
removing random factors from the tests.
"""

import pytest
import numpy as np
from src.coordination.gnn_coordinator import GNNCoordinator

# --- Initialization Tests (Still Valid) ---

def test_gnncoordinator_init_valid():
    """Test valid GNNCoordinator initialization."""
    gnn = GNNCoordinator(spectral_norm_bound=0.8, temperature=1.5)
    assert gnn.l_total_bound == 0.8
    assert gnn.temperature == 1.5

def test_gnncoordinator_init_invalid():
    """Test that invalid spectral norm bounds raise ValueError."""
    with pytest.raises(ValueError, match="spectral_norm_bound must be between 0 and 1"):
        GNNCoordinator(spectral_norm_bound=0)

# --- Updated and Robust Synergy Tests ---

def test_assign_tasks_is_influenced_by_aco_pheromones():
    """Verify that a strong pheromone signal correctly sways the GNN assignment."""
    gnn = GNNCoordinator(seed=1) # Seed doesn't matter here, but good practice
    # In this scenario, ACO's historical data strongly favors AgentB.
    bio_signals_favor_b = {
        "pso_global_best": np.zeros(gnn.embedding_dim),  # Neutral PSO signal
        "pheromone_levels": {
            "(sentiment, AgentA)": 0.1,
            "(sentiment, AgentB)": 0.9  # Very strong signal for AgentB
        },
        "conflict_weights": (0.5, 0.5)
    }
    
    assignments = gnn.assign_tasks(
        tasks=["sentiment"], 
        agents=["AgentA", "AgentB"], 
        bio_signals=bio_signals_favor_b
    )
    # The GNN should follow the strong historical signal from ACO.
    assert assignments["sentiment"] == "AgentB"


def test_assign_tasks_is_influenced_by_pso_global_best(monkeypatch):
    """
    Verify PSO influence using controlled, orthogonal embeddings to remove randomness.
    """
    gnn = GNNCoordinator()
    agents = ["AgentA", "AgentB"]
    tasks = ["data_processing"]

    # 1. Create hand-crafted, orthogonal embeddings.
    mock_embeddings = {
        "AgentA": np.array([1.0, 0.0]),
        "AgentB": np.array([0.0, 1.0]),
        "data_processing": np.array([0.5, 0.5])
    }
    
    # 2. Use monkeypatch to replace the real embedding function with our mock one.
    monkeypatch.setattr(gnn, '_get_base_embeddings', lambda nodes: mock_embeddings)

    # 3. Craft g_best to be identical to AgentA's embedding.
    g_best_for_a = mock_embeddings["AgentA"]
    
    bio_signals = {
        "pso_global_best": g_best_for_a,
        "pheromone_levels": {},          # No ACO influence
        "conflict_weights": (0.9, 0.1)   # Strongly favor PSO
    }
    
    assignments = gnn.assign_tasks(tasks=tasks, agents=agents, bio_signals=bio_signals)
    
    # Logic: AgentA's pso_score will be 1.0; AgentB's will be 0.0. AgentA must win.
    assert assignments["data_processing"] == "AgentA"


def test_assign_tasks_abc_weights_resolve_conflict(monkeypatch):
    """
    Verify ABC weight logic using controlled, orthogonal embeddings.
    """
    gnn = GNNCoordinator()
    agents = ["AgentA", "AgentB"]
    tasks = ["analysis"]

    # 1. Create hand-crafted, orthogonal embeddings.
    mock_embeddings = {
        "AgentA": np.array([1.0, 0.0]),
        "AgentB": np.array([0.0, 1.0]),
        "analysis": np.array([0.5, 0.5])
    }
    
    monkeypatch.setattr(gnn, '_get_base_embeddings', lambda nodes: mock_embeddings)

    # 2. Setup CONFLICT: PSO signal favors AgentA, ACO signal favors AgentB.
    pso_signal = mock_embeddings["AgentA"]
    aco_signal = {"(analysis, AgentB)": 0.9}

    # Case 1: ABC decides to trust ACO's historical data (high ACO weight)
    bio_signals_favor_aco = {
        "pso_global_best": pso_signal, "pheromone_levels": aco_signal, "conflict_weights": (0.1, 0.9)
    }
    # Expected scores: ScoreA=0.1*1 + 0.9*0 = 0.1; ScoreB=0.1*0 + 0.9*0.9 = 0.81. B must win.
    assignments1 = gnn.assign_tasks(tasks=tasks, agents=agents, bio_signals=bio_signals_favor_aco)
    assert assignments1["analysis"] == "AgentB"
    
    # Case 2: ABC decides to trust PSO's current tactical data (high PSO weight)
    bio_signals_favor_pso = {
        "pso_global_best": pso_signal, "pheromone_levels": aco_signal, "conflict_weights": (0.9, 0.1)
    }
    # Expected scores: ScoreA=0.9*1 + 0.1*0 = 0.9; ScoreB=0.9*0 + 0.1*0.9 = 0.09. A must win.
    assignments2 = gnn.assign_tasks(tasks=tasks, agents=agents, bio_signals=bio_signals_favor_pso)
    assert assignments2["analysis"] == "AgentA"