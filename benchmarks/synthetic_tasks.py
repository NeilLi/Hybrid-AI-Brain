#!/usr/bin/env python3
"""
benchmarks/synthetic_tasks.py

Generates synthetic TaskGraph instances for benchmarks and performance evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import networkx as nx
from typing import List

# Ensure the src directory is in the path for import resolution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core import LegacyTaskGraph as TaskGraph


def generate_synthetic_task_graph(
    num_subtasks: int,
    capability_dim: int,
    dependency_prob: float = 0.3,
    random_seed: int | None = None,
) -> TaskGraph:
    """Generate a valid random TaskGraph (DAG) for benchmarking.

    Args:
        num_subtasks: Number of subtask nodes in the graph.
        capability_dim: Dimensionality of required capability vectors.
        dependency_prob: Probability of creating a dependency between any two tasks.
        random_seed: Optional. If set, ensures reproducible graphs.

    Returns:
        A TaskGraph instance with random nodes and edges (DAG).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    task_graph = TaskGraph()
    task_ids = [f"subtask_{i}" for i in range(num_subtasks)]

    # 1. Add subtask nodes with random capability requirements
    for task_id in task_ids:
        required_capabilities = np.random.rand(capability_dim)
        task_graph.add_subtask(task_id, required_capabilities)

    # 2. Add forward-only dependencies to guarantee acyclicity
    for i in range(num_subtasks):
        for j in range(i + 1, num_subtasks):
            if np.random.rand() < dependency_prob:
                task_graph.add_dependency(task_ids[i], task_ids[j])

    return task_graph


def task_graph_stats(task_graph: TaskGraph) -> dict:
    """Return simple statistics for a TaskGraph."""
    num_nodes = len(task_graph.get_all_subtasks())
    num_edges = len(task_graph.get_dependencies())
    density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
    }


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    print("====== Benchmarks: Synthetic Task Generator Demo ======")

    # Default parameters for quick smoke test
    params = {
        "num_subtasks": 5,
        "capability_dim": 10,
        "dependency_prob": 0.4,
        "random_seed": 42,
    }

    print("\n--- Generating a synthetic task graph with parameters: ---")
    print(params)

    synthetic_task = generate_synthetic_task_graph(**params)

    print("\n--- Task Graph Statistics ---")
    stats = task_graph_stats(synthetic_task)
    for k, v in stats.items():
        print(f"{k:>12}: {v}")

    print("\n--- Edge List ---")
    for u, v in synthetic_task.get_dependencies():
        print(f"{u} -> {v}")

    print("\nSynthetic task graph generation completed successfully! 🎉")


if __name__ == "__main__":
    main()
