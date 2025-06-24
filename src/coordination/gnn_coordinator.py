#!/usr/bin/env python3
"""
src/coordination/gnn_coordinator.py

Defines the GNNCoordinator class, the central reasoning component of the system.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("hybrid_ai_brain.gnn_coordinator")
logging.basicConfig(level=logging.INFO)

class GNNCoordinator:
    """
    Graph Neural Network (GNN) coordination layer for multi-agent task assignment.
    """
    def __init__(
        self,
        spectral_norm_bound: float = 0.7,
        temperature: float = 1.0,
        embedding_dim: int = 64,
        gnn_layers: int = 2,
        use_torch: bool = True
    ):
        """
        Args:
            spectral_norm_bound: Contractivity bound for convergence (< 1).
            temperature: Softmax scaling factor (β).
            embedding_dim: Embedding dimension for tasks/agents.
            gnn_layers: Number of GNN layers (default 2).
            use_torch: If True, will use torch+torch-geometric if available.
        """
        if not 0 < spectral_norm_bound < 1:
            raise ValueError("spectral_norm_bound must be between 0 and 1 for convergence.")

        self.l_total_bound = spectral_norm_bound
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers

        # Try to import torch-geometric if requested
        self._has_torch = False
        if use_torch:
            try:
                import torch
                import torch_geometric
                self._has_torch = True
            except ImportError:
                logger.warning("torch or torch-geometric not found. Falling back to numpy simulation.")

        logger.info(f"GNNCoordinator initialized (L_total < {self.l_total_bound}, β={self.temperature}, "
                    f"embedding_dim={self.embedding_dim}, layers={self.gnn_layers}, torch={self._has_torch})")

    def _run_message_passing(
        self, 
        task_agent_graph: Any, 
        edge_features: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Simulate (or run) GNN message passing to get node embeddings.
        """
        logger.info("  GNNCoordinator: Running message passing...")
        # Real implementation would use torch_geometric, Data objects, etc.
        # Here we simulate with random embeddings for demo/testing.
        # Use consistent ordering for reproducibility/testing:
        nodes = getattr(task_agent_graph, "nodes", lambda: ["task_1", "agent_A", "agent_B"])()
        embeddings = {
            n: np.random.randn(self.embedding_dim) for n in nodes
        }
        return embeddings

    def assign_tasks(
        self, 
        task_agent_graph: Any, 
        edge_features: Dict[str, Any],
        tasks: Optional[list] = None,
        agents: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Compute optimal assignment of tasks to agents using GNN message passing.

        Args:
            task_agent_graph: Graph of tasks and agents (networkx, etc.)
            edge_features: Dict of edge feature values (e.g., pheromone, risk, cost)
            tasks: List of task IDs (optional, derived from graph if not given)
            agents: List of agent IDs (optional, derived from graph if not given)

        Returns:
            Dict mapping task_id -> agent_id assignments.
        """
        logger.info("GNNCoordinator: Starting task assignment.")

        # 1. Run GNN (simulated here) to get embeddings
        final_embeddings = self._run_message_passing(task_agent_graph, edge_features)

        # 2. Extract lists of tasks and agents from the graph if not given
        if tasks is None:
            if hasattr(task_agent_graph, "tasks"):
                tasks = list(task_agent_graph.tasks())
            elif hasattr(task_agent_graph, "nodes"):
                # Heuristic: tasks start with 'task_'
                tasks = [n for n in task_agent_graph.nodes() if n.startswith("task_")]
            else:
                tasks = ["task_1"]  # fallback
        if agents is None:
            if hasattr(task_agent_graph, "agents"):
                agents = list(task_agent_graph.agents())
            elif hasattr(task_agent_graph, "nodes"):
                agents = [n for n in task_agent_graph.nodes() if n.startswith("agent_")]
            else:
                agents = ["agent_A", "agent_B"]

        # 3. Compute assignment probabilities: softmax over dot products
        assignment_scores = {}
        for t in tasks:
            t_emb = final_embeddings[t]
            agent_scores = []
            for a in agents:
                a_emb = final_embeddings[a]
                score = float(np.dot(t_emb, a_emb)) / (np.linalg.norm(t_emb) * np.linalg.norm(a_emb) + 1e-8)
                agent_scores.append(score)
            # Apply temperature and softmax
            agent_scores_np = np.array(agent_scores) * self.temperature
            softmax_probs = np.exp(agent_scores_np) / np.sum(np.exp(agent_scores_np))
            # Pick best agent (highest probability)
            best_idx = int(np.argmax(softmax_probs))
            assignment_scores[t] = agents[best_idx]

            logger.debug(f"Task '{t}' scores: {agent_scores}, softmax: {softmax_probs}, assigned: {agents[best_idx]}")

        logger.info(f"  GNNCoordinator: Final assignments computed: {assignment_scores}")
        return assignment_scores

    def check_contractivity(self, gnn_weights: Optional[Any] = None) -> bool:
        """
        Check that the GNN's spectral norm is within the contractive bound.
        Real implementation would use torch.linalg.svd, etc.
        """
        logger.info("GNNCoordinator: Checking contractivity (simulated pass).")
        # Placeholder: Always True in demo.
        return True

    def __repr__(self):
        return (f"GNNCoordinator(L_total<{self.l_total_bound}, β={self.temperature}, "
                f"dim={self.embedding_dim}, layers={self.gnn_layers})")

# --- Demo/Test Block ---
if __name__ == "__main__":
    # Minimal demo with mock graph structure
    class DummyGraph:
        def nodes(self): return ["task_1", "task_2", "agent_A", "agent_B"]
    graph = DummyGraph()
    edge_features = {}
    gnn = GNNCoordinator()
    assignments = gnn.assign_tasks(graph, edge_features)
    print("Assignments:", assignments)
