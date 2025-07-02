#!/usr/bin/env python3
"""
src/coordination/gnn_coordinator.py

Implements the GNN Coordinator with Bio-GNN Coordination Protocol,
integrating bio-inspired optimization signals with graph neural network
coordination as described in the "Hybrid AI Brain" paper.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

# Import enhanced bio optimizer
from enhanced_bio_optimizer import EnhancedBioOptimizer
from conflict_resolver import ConflictResolver

logger = logging.getLogger("hybrid_ai_brain.gnn_coordinator")

@dataclass
class AgentNode:
    """Represents an agent as a node in the coordination graph."""
    agent_id: str
    capabilities: Dict[str, float]
    current_load: float = 0.0
    role: str = "Onlooker"  # ABC role: Employed, Onlooker, Scout
    embedding: Optional[np.ndarray] = None

@dataclass
class TaskNode:
    """Represents a task as a node in the coordination graph."""
    task_id: str
    requirements: Dict[str, float]
    priority: float = 1.0
    deadline: Optional[float] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class CoordinationEdge:
    """Represents an edge in the coordination graph."""
    source_id: str
    target_id: str
    weight: float
    edge_type: str  # 'agent_task', 'agent_agent', 'task_task'
    pheromone_level: float = 0.0

class GNNLayer:
    """Single layer of the Graph Neural Network."""
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Initialize weights with Xavier initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
        self.lipschitz_bound = 0.9  # Ensure contraction property
        
        # Normalize weights to satisfy Lipschitz constraint
        self._enforce_lipschitz_constraint()
    
    def _enforce_lipschitz_constraint(self):
        """Ensure L_total < 1 for convergence guarantee."""
        spectral_norm = np.linalg.norm(self.W, ord=2)
        if spectral_norm >= self.lipschitz_bound:
            self.W = self.W * (self.lipschitz_bound / spectral_norm)
    
    def forward(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Forward pass through GNN layer."""
        # Message passing: aggregate neighbor features
        messages = np.dot(adjacency, node_features)
        
        # Apply linear transformation
        output = np.dot(messages, self.W) + self.b
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'tanh':
            output = np.tanh(output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-np.clip(output, -500, 500)))
        
        return output
    
    def update_weights(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """Update layer weights with gradient and enforce constraints."""
        self.W -= learning_rate * gradient
        self._enforce_lipschitz_constraint()

class BiGNNCoordinator:
    """Graph Neural Network Coordinator with Bio-inspired Protocol Integration."""
    
    def __init__(self, embedding_dim: int = 64, num_layers: int = 3, 
                 delta_bio: float = 2.0, delta_gnn: float = 0.2):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.delta_bio = delta_bio  # Bio-layer update interval (2s)
        self.delta_gnn = delta_gnn  # GNN coordination interval (0.2s)
        
        # Initialize GNN layers
        self.layers = []
        layer_dims = [embedding_dim] + [embedding_dim] * (num_layers - 1) + [embedding_dim]
        for i in range(num_layers):
            layer = GNNLayer(layer_dims[i], layer_dims[i + 1])
            self.layers.append(layer)
        
        # Coordination graph components
        self.agent_nodes: Dict[str, AgentNode] = {}
        self.task_nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[Tuple[str, str], CoordinationEdge] = {}
        
        # Bio-inspired optimization
        self.bio_optimizer = EnhancedBioOptimizer()
        self.conflict_resolver = ConflictResolver()
        
        # Timing and state
        self.last_bio_update = 0.0
        self.gnn_iteration = 0
        self.coordination_history = []
        
        # Safety constraints
        self.safety_threshold = 0.7
        
        logger.info(f"BiGNNCoordinator initialized with {num_layers} layers, "
                   f"bio_interval={delta_bio}s, gnn_interval={delta_gnn}s")
    
    def add_agent(self, agent_id: str, capabilities: Dict[str, float]):
        """Add an agent node to the coordination graph."""
        embedding = self._generate_embedding(capabilities)
        agent = AgentNode(agent_id=agent_id, capabilities=capabilities, embedding=embedding)
        self.agent_nodes[agent_id] = agent
        logger.info(f"Added agent {agent_id} with capabilities: {capabilities}")
    
    def add_task(self, task_id: str, requirements: Dict[str, float], priority: float = 1.0):
        """Add a task node to the coordination graph."""
        embedding = self._generate_embedding(requirements)
        task = TaskNode(task_id=task_id, requirements=requirements, 
                       priority=priority, embedding=embedding)
        self.task_nodes[task_id] = task
        logger.info(f"Added task {task_id} with requirements: {requirements}")
    
    def _generate_embedding(self, features: Dict[str, float]) -> np.ndarray:
        """Generate node embedding from feature dictionary."""
        # Convert feature dict to fixed-size vector
        feature_keys = ['sentiment_analysis', 'multilingual', 'reasoning', 'creativity', 'priority']
        embedding = np.zeros(self.embedding_dim)
        
        for i, key in enumerate(feature_keys[:min(len(feature_keys), self.embedding_dim)]):
            if key in features:
                embedding[i] = features[key]
        
        # Fill remaining dimensions with derived features
        if len(features) > 0:
            for i in range(len(feature_keys), self.embedding_dim):
                embedding[i] = np.random.normal(0, 0.1)  # Small random component
        
        return embedding
    
    def _build_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Build adjacency matrix from coordination graph."""
        all_nodes = list(self.agent_nodes.keys()) + list(self.task_nodes.keys())
        n_nodes = len(all_nodes)
        
        if n_nodes == 0:
            return np.array([]), []
        
        adjacency = np.zeros((n_nodes, n_nodes))
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        # Add edges from coordination graph
        for (source, target), edge in self.edges.items():
            if source in node_to_idx and target in node_to_idx:
                i, j = node_to_idx[source], node_to_idx[target]
                adjacency[i, j] = edge.weight
                
                # For undirected edges (agent-agent, task-task), make symmetric
                if edge.edge_type in ['agent_agent', 'task_task']:
                    adjacency[j, i] = edge.weight
        
        return adjacency, all_nodes
    
    def _get_node_features(self, node_list: List[str]) -> np.ndarray:
        """Extract node features for GNN processing."""
        features = []
        for node_id in node_list:
            if node_id in self.agent_nodes:
                embedding = self.agent_nodes[node_id].embedding
            elif node_id in self.task_nodes:
                embedding = self.task_nodes[node_id].embedding
            else:
                embedding = np.zeros(self.embedding_dim)
            features.append(embedding)
        
        return np.array(features) if features else np.array([]).reshape(0, self.embedding_dim)
    
    def _update_edge_weights_from_bio(self, bio_result: Dict[str, Any]):
        """Update edge weights based on bio-inspired optimization results."""
        logger.info("GNN: Updating edge weights from bio-optimization results...")
        
        # Extract bio-optimization signals
        pso_global_best = bio_result.get("pso_global_best", np.array([]))
        pheromone_levels = bio_result.get("pheromone_levels", {})
        conflict_weights = bio_result.get("conflict_weights", (0.5, 0.5))
        role_assignments = bio_result.get("role_assignments", {})
        
        lambda_pso, lambda_aco = conflict_weights
        
        # Update agent roles from ABC
        for agent_id, role in role_assignments.items():
            if agent_id in self.agent_nodes:
                self.agent_nodes[agent_id].role = role
        
        # Generate PSO-based edge weights
        pso_weights = {}
        if len(pso_global_best) > 0:
            for i, (edge_key, edge) in enumerate(self.edges.items()):
                if i < len(pso_global_best):
                    pso_weights[edge_key] = float(pso_global_best[i])
                else:
                    pso_weights[edge_key] = 0.5  # Default weight
        
        # Generate ACO-based edge weights from pheromone levels
        aco_weights = {}
        for edge_key, edge in self.edges.items():
            source, target = edge_key
            path_key = f"({edge.edge_type}, {source}, {target})"
            pheromone = pheromone_levels.get(path_key, 0.1)
            edge.pheromone_level = pheromone
            aco_weights[edge_key] = pheromone
        
        # Resolve conflicts using ABC weights (Equation 9 from paper)
        final_weights = self.conflict_resolver.resolve(pso_weights, aco_weights, conflict_weights)
        
        # Apply resolved weights with safety validation
        safety_valid = True
        for edge_key, weight in final_weights.items():
            if edge_key in self.edges:
                # Safety constraint validation
                if weight < 0 or weight > 1:
                    weight = np.clip(weight, 0, 1)
                    safety_valid = False
                
                self.edges[edge_key].weight = weight
        
        if not safety_valid:
            logger.warning("Edge weights clipped to maintain safety constraints")
        
        return final_weights
    
    def _compute_task_agent_compatibility(self, task_id: str, agent_id: str) -> float:
        """Compute compatibility score between task and agent."""
        if task_id not in self.task_nodes or agent_id not in self.agent_nodes:
            return 0.0
        
        task = self.task_nodes[task_id]
        agent = self.agent_nodes[agent_id]
        
        # Calculate capability matching score
        total_score = 0.0
        total_weight = 0.0
        
        for capability, required_level in task.requirements.items():
            if capability in agent.capabilities:
                agent_level = agent.capabilities[capability]
                score = min(agent_level / required_level, 1.0) if required_level > 0 else 1.0
                total_score += score * required_level
                total_weight += required_level
        
        compatibility = total_score / total_weight if total_weight > 0 else 0.0
        
        # Apply role-based modifiers from ABC
        role_modifier = 1.0
        if agent.role == 'Employed':
            role_modifier = 1.2  # Employed agents get preference
        elif agent.role == 'Scout':
            role_modifier = 0.8  # Scouts are exploring, less preferred for current tasks
        
        return min(1.0, compatibility * role_modifier)
    
    def _gnn_forward_pass(self, node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        """Execute forward pass through all GNN layers."""
        current_features = node_features
        
        for i, layer in enumerate(self.layers):
            current_features = layer.forward(current_features, adjacency)
            
            # Apply residual connection for deeper networks
            if i > 0 and current_features.shape == node_features.shape:
                current_features = current_features + 0.1 * node_features
        
        return current_features
    
    def _generate_task_assignments(self, updated_features: np.ndarray, 
                                 node_list: List[str]) -> Dict[str, str]:
        """Generate task assignments from updated node features."""
        assignments = {}
        
        # Separate agents and tasks
        agent_indices = [i for i, node in enumerate(node_list) if node in self.agent_nodes]
        task_indices = [i for i, node in enumerate(node_list) if node in self.task_nodes]
        
        if not agent_indices or not task_indices:
            return assignments
        
        # Compute assignment scores using updated embeddings
        for task_idx in task_indices:
            task_id = node_list[task_idx]
            task_embedding = updated_features[task_idx]
            
            best_agent = None
            best_score = -1.0
            
            for agent_idx in agent_indices:
                agent_id = node_list[agent_idx]
                agent_embedding = updated_features[agent_idx]
                
                # Compute similarity score
                similarity = np.dot(task_embedding, agent_embedding) / (
                    np.linalg.norm(task_embedding) * np.linalg.norm(agent_embedding) + 1e-8
                )
                
                # Add compatibility bonus
                compatibility = self._compute_task_agent_compatibility(task_id, agent_id)
                final_score = 0.7 * similarity + 0.3 * compatibility
                
                # Consider agent load (prefer less loaded agents)
                if agent_id in self.agent_nodes:
                    load_penalty = self.agent_nodes[agent_id].current_load * 0.2
                    final_score -= load_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_agent = agent_id
            
            if best_agent and best_score > 0.3:  # Minimum threshold
                assignments[task_id] = best_agent
                
                # Update agent load
                if best_agent in self.agent_nodes:
                    self.agent_nodes[best_agent].current_load += 0.1
        
        return assignments
    
    def coordinate(self, current_time: float) -> Dict[str, Any]:
        """Execute one coordination cycle with bio-inspired optimization."""
        self.gnn_iteration += 1
        logger.info(f"=== GNN Coordination Cycle {self.gnn_iteration} (t={current_time:.1f}s) ===")
        
        # Check if bio-optimization update is needed
        bio_result = None
        if current_time - self.last_bio_update >= self.delta_bio:
            logger.info("Bio-optimization update triggered...")
            
            # Prepare system state for bio-optimization
            agent_fitness = {}
            for agent_id, agent in self.agent_nodes.items():
                # Calculate fitness based on capability strength and current role
                capability_sum = sum(agent.capabilities.values())
                load_factor = max(0.1, 1.0 - agent.current_load)
                agent_fitness[agent_id] = capability_sum * load_factor
            
            successful_paths = {}
            for (source, target), edge in self.edges.items():
                if edge.edge_type == 'agent_task' and edge.weight > 0.5:
                    path_key = f"({edge.edge_type}, {source}, {target})"
                    successful_paths[path_key] = edge.weight * edge.pheromone_level
            
            system_state = {
                "agent_fitness": agent_fitness,
                "successful_paths": successful_paths,
                "context": "multilingual" if self.gnn_iteration % 5 == 0 else "default",
                "num_active_tasks": len(self.task_nodes)
            }
            
            # Run bio-optimization
            bio_result = self.bio_optimizer.run_optimization_cycle(system_state)
            
            # Update edge weights from bio-optimization
            self._update_edge_weights_from_bio(bio_result)
            self.last_bio_update = current_time
        
        # Build graph representation
        adjacency, node_list = self._build_adjacency_matrix()
        
        if len(node_list) == 0:
            logger.warning("No nodes in coordination graph")
            return {"assignments": {}, "bio_result": bio_result}
        
        # Get node features
        node_features = self._get_node_features(node_list)
        
        # Execute GNN forward pass
        updated_features = self._gnn_forward_pass(node_features, adjacency)
        
        # Generate task assignments
        assignments = self._generate_task_assignments(updated_features, node_list)
        
        # Calculate coordination quality metrics
        coordination_quality = self._calculate_coordination_quality(assignments)
        
        # Store coordination history
        coordination_record = {
            "iteration": self.gnn_iteration,
            "time": current_time,
            "assignments": assignments,
            "quality": coordination_quality,
            "bio_updated": bio_result is not None
        }
        self.coordination_history.append(coordination_record)
        
        logger.info(f"Coordination complete: {len(assignments)} assignments, quality={coordination_quality:.3f}")
        
        return {
            "assignments": assignments,
            "coordination_quality": coordination_quality,
            "bio_result": bio_result,
            "node_features": updated_features,
            "adjacency": adjacency,
            "safety_threshold_met": coordination_quality >= self.safety_threshold
        }
    
    def _calculate_coordination_quality(self, assignments: Dict[str, str]) -> float:
        """Calculate overall coordination quality score."""
        if not assignments:
            return 0.0
        
        total_quality = 0.0
        for task_id, agent_id in assignments.items():
            compatibility = self._compute_task_agent_compatibility(task_id, agent_id)
            total_quality += compatibility
        
        return total_quality / len(assignments)
    
    def update_agent_load(self, agent_id: str, load_delta: float):
        """Update agent load (e.g., when task completes)."""
        if agent_id in self.agent_nodes:
            old_load = self.agent_nodes[agent_id].current_load
            self.agent_nodes[agent_id].current_load = max(0.0, old_load + load_delta)
            logger.info(f"Agent {agent_id} load: {old_load:.2f} -> {self.agent_nodes[agent_id].current_load:.2f}")
    
    def add_coordination_edge(self, source: str, target: str, edge_type: str, initial_weight: float = 0.5):
        """Add an edge to the coordination graph."""
        edge = CoordinationEdge(
            source_id=source,
            target_id=target,
            weight=initial_weight,
            edge_type=edge_type
        )
        self.edges[(source, target)] = edge
        logger.info(f"Added {edge_type} edge: {source} -> {target} (weight={initial_weight})")
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics and performance metrics."""
        if not self.coordination_history:
            return {}
        
        recent_history = self.coordination_history[-10:]  # Last 10 cycles
        avg_quality = np.mean([record["quality"] for record in recent_history])
        
        role_distribution = defaultdict(int)
        for agent in self.agent_nodes.values():
            role_distribution[agent.role] += 1
        
        return {
            "total_cycles": len(self.coordination_history),
            "average_quality": avg_quality,
            "current_agents": len(self.agent_nodes),
            "current_tasks": len(self.task_nodes),
            "role_distribution": dict(role_distribution),
            "total_edges": len(self.edges)
        }

# Demo and Testing
if __name__ == "__main__":
    print("=" * 80)
    print("DEMO: BiGNN Coordinator with Bio-Inspired Protocol")
    print("=" * 80)
    
    # Initialize coordinator
    coordinator = BiGNNCoordinator(embedding_dim=32, num_layers=2)
    
    # Add agents with different capabilities
    agents = {
        "Agent_A": {"sentiment_analysis": 0.9, "multilingual": 0.8, "reasoning": 0.7},
        "Agent_B": {"sentiment_analysis": 0.6, "multilingual": 0.9, "reasoning": 0.8},
        "Agent_C": {"sentiment_analysis": 0.8, "multilingual": 0.5, "reasoning": 0.9},
        "Agent_D": {"sentiment_analysis": 0.7, "multilingual": 0.7, "reasoning": 0.6}
    }
    
    for agent_id, capabilities in agents.items():
        coordinator.add_agent(agent_id, capabilities)
    
    # Add tasks with different requirements
    tasks = {
        "Task_Sentiment": {"sentiment_analysis": 0.8, "multilingual": 0.3},
        "Task_Multilingual": {"sentiment_analysis": 0.4, "multilingual": 0.9},
        "Task_Reasoning": {"sentiment_analysis": 0.2, "multilingual": 0.2, "reasoning": 0.8}
    }
    
    for task_id, requirements in tasks.items():
        coordinator.add_task(task_id, requirements)
    
    # Add coordination edges
    for agent_id in agents.keys():
        for task_id in tasks.keys():
            coordinator.add_coordination_edge(agent_id, task_id, "agent_task")
    
    # Run multiple coordination cycles
    print("\n[Running Coordination Cycles]\n")
    
    for cycle in range(5):
        current_time = cycle * 0.2  # GNN runs every 0.2s
        result = coordinator.coordinate(current_time)
        
        print(f"Cycle {cycle + 1} (t={current_time:.1f}s):")
        print(f"  Assignments: {result['assignments']}")
        print(f"  Quality: {result['coordination_quality']:.3f}")
        print(f"  Bio-updated: {result.get('bio_result') is not None}")
        print()
        
        # Simulate some task completions
        if cycle == 2:
            coordinator.update_agent_load("Agent_A", -0.3)  # Task completed
            coordinator.update_agent_load("Agent_B", 0.2)   # New task assigned
    
    # Show final statistics
    print("==> Final Coordination Statistics:")
    stats = coordinator.get_coordination_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)