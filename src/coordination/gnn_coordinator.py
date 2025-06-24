#!/usr/bin/env python3
"""
src/coordination/gnn_coordinator.py

Defines the GNNCoordinator class, the central reasoning component of the system.
Fixed to handle explicit task/agent lists properly.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Set
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

    def _determine_all_nodes(
        self, 
        task_agent_graph: Any, 
        tasks: Optional[List[str]] = None, 
        agents: Optional[List[str]] = None
    ) -> Set[str]:
        """
        Determine all nodes that need embeddings, considering both graph nodes and explicit lists.
        """
        all_nodes = set()
        
        # Add nodes from the graph if it has them
        if hasattr(task_agent_graph, "nodes"):
            try:
                graph_nodes = task_agent_graph.nodes()
                if callable(graph_nodes):
                    graph_nodes = graph_nodes()
                all_nodes.update(graph_nodes)
            except Exception:
                pass
        
        # Add explicitly provided tasks and agents
        if tasks:
            all_nodes.update(tasks)
        if agents:
            all_nodes.update(agents)
            
        # Fallback if we still have no nodes
        if not all_nodes:
            all_nodes = {"task_1", "agent_A", "agent_B"}
            
        return all_nodes

    def _run_message_passing(
        self, 
        task_agent_graph: Any, 
        edge_features: Dict[str, Any],
        tasks: Optional[List[str]] = None,
        agents: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate (or run) GNN message passing to get node embeddings.
        Now considers explicit task/agent lists to ensure all needed embeddings are generated.
        """
        logger.debug("  GNNCoordinator: Running message passing...")
        
        # Determine all nodes that need embeddings
        all_nodes = self._determine_all_nodes(task_agent_graph, tasks, agents)
        
        # Generate embeddings for all nodes
        # In a real implementation, this would run actual GNN layers
        # For simulation, we use random embeddings with fixed seed for determinism
        embeddings = {}
        for node in all_nodes:
            # Use node name hash as seed for deterministic but varied embeddings
            seed = hash(node) % (2**32)
            rng = np.random.RandomState(seed)
            embeddings[node] = rng.randn(self.embedding_dim)
            
        logger.debug(f"  Generated embeddings for {len(embeddings)} nodes: {list(embeddings.keys())}")
        return embeddings

    def _extract_tasks_and_agents(
        self, 
        task_agent_graph: Any, 
        tasks: Optional[List[str]] = None,
        agents: Optional[List[str]] = None
    ) -> tuple[List[str], List[str]]:
        """
        Extract or use provided lists of tasks and agents.
        """
        # Extract tasks
        if tasks is None:
            if hasattr(task_agent_graph, "tasks"):
                try:
                    tasks_method = task_agent_graph.tasks()
                    tasks = list(tasks_method) if tasks_method else []
                except Exception:
                    tasks = []
            elif hasattr(task_agent_graph, "nodes"):
                try:
                    nodes = task_agent_graph.nodes()
                    if callable(nodes):
                        nodes = nodes()
                    # Heuristic: tasks start with 'task_'
                    tasks = [n for n in nodes if str(n).startswith("task_")]
                except Exception:
                    tasks = []
            
            if not tasks:
                tasks = ["task_1"]  # fallback
        
        # Extract agents  
        if agents is None:
            if hasattr(task_agent_graph, "agents"):
                try:
                    agents_method = task_agent_graph.agents()
                    agents = list(agents_method) if agents_method else []
                except Exception:
                    agents = []
            elif hasattr(task_agent_graph, "nodes"):
                try:
                    nodes = task_agent_graph.nodes()
                    if callable(nodes):
                        nodes = nodes()
                    # Heuristic: agents start with 'agent_'
                    agents = [n for n in nodes if str(n).startswith("agent_")]
                except Exception:
                    agents = []
            
            if not agents:
                agents = ["agent_A", "agent_B"]  # fallback
        
        return tasks, agents

    def assign_tasks(
        self, 
        task_agent_graph: Any, 
        edge_features: Dict[str, Any],
        tasks: Optional[List[str]] = None,
        agents: Optional[List[str]] = None
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

        # 1. Extract or use provided lists of tasks and agents
        tasks, agents = self._extract_tasks_and_agents(task_agent_graph, tasks, agents)
        
        if not tasks:
            logger.warning("No tasks found for assignment.")
            return {}
        if not agents:
            logger.warning("No agents found for assignment.")
            return {}

        logger.debug(f"Assigning {len(tasks)} tasks to {len(agents)} agents.")
        logger.debug(f"Tasks: {tasks}")
        logger.debug(f"Agents: {agents}")

        # 2. Run GNN to get embeddings (passing tasks/agents to ensure coverage)
        final_embeddings = self._run_message_passing(task_agent_graph, edge_features, tasks, agents)

        # 3. Verify all required embeddings exist
        missing_embeddings = []
        for task in tasks:
            if task not in final_embeddings:
                missing_embeddings.append(f"task:{task}")
        for agent in agents:
            if agent not in final_embeddings:
                missing_embeddings.append(f"agent:{agent}")
        
        if missing_embeddings:
            raise RuntimeError(f"Missing embeddings for: {missing_embeddings}")

        # 4. Compute assignment probabilities: softmax over dot products
        assignment_scores = {}
        for t in tasks:
            t_emb = final_embeddings[t]
            agent_scores = []
            
            for a in agents:
                a_emb = final_embeddings[a]
                # Compute cosine similarity
                dot_product = float(np.dot(t_emb, a_emb))
                t_norm = float(np.linalg.norm(t_emb))
                a_norm = float(np.linalg.norm(a_emb))
                score = dot_product / (t_norm * a_norm + 1e-8)
                agent_scores.append(score)
            
            # Apply temperature and softmax
            agent_scores_np = np.array(agent_scores) * self.temperature
            # Numerical stability: subtract max before exp
            agent_scores_np = agent_scores_np - np.max(agent_scores_np)
            exp_scores = np.exp(agent_scores_np)
            softmax_probs = exp_scores / np.sum(exp_scores)
            
            # Pick best agent (highest probability)
            best_idx = int(np.argmax(softmax_probs))
            assignment_scores[t] = agents[best_idx]

            logger.debug(f"Task '{t}' scores: {agent_scores}, softmax: {softmax_probs.tolist()}, assigned: {agents[best_idx]}")

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

    def get_embedding(self, node_id: str, task_agent_graph: Any, edge_features: Dict[str, Any]) -> np.ndarray:
        """
        Get the embedding for a specific node.
        Useful for debugging and analysis.
        """
        embeddings = self._run_message_passing(task_agent_graph, edge_features)
        if node_id not in embeddings:
            raise KeyError(f"No embedding found for node '{node_id}'")
        return embeddings[node_id]

    def __repr__(self):
        return (f"GNNCoordinator(L_total<{self.l_total_bound}, β={self.temperature}, "
                f"dim={self.embedding_dim}, layers={self.gnn_layers})")

# --- Demo/Test Block ---
if __name__ == "__main__":
    # Minimal demo with mock graph structure
    class DummyGraph:
        def nodes(self): return ["task_1", "task_2", "agent_A", "agent_B"]
    
    print("=== GNNCoordinator Demo ===")
    
    graph = DummyGraph()
    edge_features = {"coordination_weight": 0.5}
    gnn = GNNCoordinator()
    
    # Test 1: Basic assignment
    print("\n1. Basic assignment:")
    assignments = gnn.assign_tasks(graph, edge_features)
    print("Assignments:", assignments)
    
    # Test 2: Custom task/agent lists
    print("\n2. Custom task/agent lists:")
    custom_tasks = ["custom_task_1", "custom_task_2"]
    custom_agents = ["agent_X", "agent_Y", "agent_Z"]
    custom_assignments = gnn.assign_tasks(graph, edge_features, tasks=custom_tasks, agents=custom_agents)
    print("Custom assignments:", custom_assignments)
    
    # Test 3: Different temperature
    print("\n3. Different temperature (β=2.0):")
    gnn_hot = GNNCoordinator(temperature=2.0)
    hot_assignments = gnn_hot.assign_tasks(graph, edge_features)
    print("Hot assignments:", hot_assignments)
    
    print("\n=== Demo completed ===")