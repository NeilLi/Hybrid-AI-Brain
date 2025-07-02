#!/usr/bin/env python3
"""
src/coordination/bio_optimizer.py

Enhanced implementation of hierarchical bio-inspired optimization (ABC -> PSO -> ACO) 
with complete algorithms and safety constraints as per the "Hybrid AI Brain" paper.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# --- Setup Logging ---
logger = logging.getLogger("hybrid_ai_brain.bio_optimizer")
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

@dataclass
class PSOParticle:
    """Represents a PSO particle with position, velocity, and fitness."""
    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    best_position: np.ndarray
    best_fitness: float

@dataclass
class ABCBee:
    """Represents an ABC bee with role and associated food source."""
    role: str  # 'Employed', 'Onlooker', or 'Scout'
    food_source: Optional[np.ndarray] = None
    fitness: float = 0.0
    trial_count: int = 0

class SafetyConstraints:
    """Enforces safety constraints from the paper."""
    
    def __init__(self, safety_threshold: float = 0.7, lipschitz_bound: float = 1.0):
        self.safety_threshold = safety_threshold
        self.lipschitz_bound = lipschitz_bound
    
    def validate_safety_threshold(self, edge_weights: Dict[str, float]) -> bool:
        """Verify τ_safe ≥ 0.7 constraint."""
        if not edge_weights:
            return True
        weights = np.array(list(edge_weights.values()))
        safety_score = np.mean(weights) * (1 - np.std(weights))
        return safety_score >= self.safety_threshold
    
    def enforce_lipschitz_constraint(self, weights: np.ndarray) -> np.ndarray:
        """Enforce L_total < 1 through spectral projection."""
        spectral_norm = np.linalg.norm(weights, ord=2)
        if spectral_norm >= self.lipschitz_bound:
            weights = weights * (self.lipschitz_bound - 1e-6) / spectral_norm
        return weights

class PSOOptimizer:
    """Particle Swarm Optimization for mesoscopic coordination."""
    
    def __init__(self, dim: int = 128, num_particles: int = 20, w: float = 0.7, 
                 c1: float = 1.5, c2: float = 1.5):
        self.dim = dim
        self.num_particles = num_particles
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.particles: List[PSOParticle] = []
        self.global_best_position = np.random.rand(dim)
        self.global_best_fitness = float('-inf')
        self.safety_constraints = SafetyConstraints()
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Initialize particle swarm."""
        for _ in range(self.num_particles):
            position = np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * 0.1
            particle = PSOParticle(
                position=position,
                velocity=velocity,
                fitness=float('-inf'),
                best_position=position.copy(),
                best_fitness=float('-inf')
            )
            self.particles.append(particle)
    
    def _evaluate_fitness(self, position: np.ndarray, agent_scores: Dict[str, float]) -> float:
        """Evaluate particle fitness based on agent coordination quality."""
        if not agent_scores:
            return np.random.random()
        
        # Convert position to agent weights
        agent_names = list(agent_scores.keys())
        num_agents = len(agent_names)
        if num_agents == 0:
            return 0.0
        
        # Map position dimensions to agent weights
        weights = position[:min(num_agents, self.dim)]
        if len(weights) < num_agents:
            weights = np.pad(weights, (0, num_agents - len(weights)), 'constant', constant_values=0.5)
        
        # Normalize weights
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(num_agents) / num_agents
        
        # Calculate fitness as weighted sum of agent scores
        fitness = sum(w * agent_scores[agent] for w, agent in zip(weights, agent_names))
        return fitness
    
    def update(self, agent_fitness_scores: Dict[str, float]) -> Dict[str, Any]:
        """Update PSO particles and return global best solution."""
        logger.info("PSO: Updating particle positions and velocities...")
        
        # Evaluate all particles
        for particle in self.particles:
            fitness = self._evaluate_fitness(particle.position, agent_fitness_scores)
            particle.fitness = fitness
            
            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
        
        # Update velocities and positions
        for particle in self.particles:
            r1, r2 = np.random.rand(2)
            
            # Velocity update with safety constraints
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social
            
            # Position update with Lipschitz constraint
            new_position = particle.position + particle.velocity
            particle.position = self.safety_constraints.enforce_lipschitz_constraint(new_position)
            particle.position = np.clip(particle.position, 0, 1)  # Keep in valid range
        
        best_agent = max(agent_fitness_scores, key=agent_fitness_scores.get) if agent_fitness_scores else None
        
        return {
            "pso_global_best": self.global_best_position,
            "best_agent": best_agent,
            "global_best_fitness": self.global_best_fitness,
            "convergence_measure": np.std([p.fitness for p in self.particles])
        }

class ACOOptimizer:
    """Ant Colony Optimization for microscopic pathfinding and memory."""
    
    def __init__(self, evaporation_rate: float = 0.5, alpha: float = 1.0, beta: float = 2.0):
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta   # Heuristic importance
        self.pheromone_map: Dict[str, float] = {}
        self.path_success_history: Dict[str, List[float]] = {}
    
    def update_pheromones(self, successful_paths: Dict[str, float]) -> Dict[str, float]:
        """Update pheromone levels based on path success."""
        logger.info("ACO: Updating pheromone trails...")
        
        # Evaporation
        for path in self.pheromone_map:
            self.pheromone_map[path] *= (1 - self.evaporation_rate)
        
        # Pheromone deposit
        for path, success_rate in successful_paths.items():
            if path not in self.pheromone_map:
                self.pheromone_map[path] = 0.1  # Initial pheromone level
            
            # Update pheromone based on success rate
            deposit = success_rate * (1 - self.evaporation_rate)
            self.pheromone_map[path] += deposit
            
            # Track success history
            if path not in self.path_success_history:
                self.path_success_history[path] = []
            self.path_success_history[path].append(success_rate)
            
            # Keep only recent history
            if len(self.path_success_history[path]) > 10:
                self.path_success_history[path] = self.path_success_history[path][-10:]
        
        # Normalize pheromone levels
        if self.pheromone_map:
            max_pheromone = max(self.pheromone_map.values())
            if max_pheromone > 0:
                for path in self.pheromone_map:
                    self.pheromone_map[path] /= max_pheromone
        
        return self.pheromone_map.copy()
    
    def get_path_probability(self, path: str, heuristic_value: float = 1.0) -> float:
        """Calculate path selection probability using ACO formula."""
        pheromone = self.pheromone_map.get(path, 0.1)
        return (pheromone ** self.alpha) * (heuristic_value ** self.beta)
    
    def get_best_historical_path(self) -> Tuple[str, float]:
        """Return the path with highest historical success."""
        if not self.path_success_history:
            return "", 0.0
        
        best_path = ""
        best_avg_success = 0.0
        
        for path, history in self.path_success_history.items():
            avg_success = np.mean(history)
            if avg_success > best_avg_success:
                best_avg_success = avg_success
                best_path = path
        
        return best_path, best_avg_success

class ABCOptimizer:
    """Artificial Bee Colony for macroscopic strategy and meta-optimization."""
    
    def __init__(self, num_employed: int = 10, num_onlookers: int = 10, 
                 num_scouts: int = 5, limit: int = 100, conflict_threshold: float = 0.05):
        self.num_employed = num_employed
        self.num_onlookers = num_onlookers
        self.num_scouts = num_scouts
        self.limit = limit  # Abandonment limit
        self.conflict_threshold = conflict_threshold
        
        self.employed_bees: List[ABCBee] = []
        self.onlooker_bees: List[ABCBee] = []
        self.scout_bees: List[ABCBee] = []
        
        self._initialize_bees()
    
    def _initialize_bees(self):
        """Initialize bee populations."""
        # Initialize employed bees with random food sources (weight combinations)
        for _ in range(self.num_employed):
            food_source = np.random.rand(2)  # [λ_PSO, λ_ACO]
            food_source = food_source / np.sum(food_source)  # Normalize
            bee = ABCBee(role='Employed', food_source=food_source)
            self.employed_bees.append(bee)
        
        # Initialize onlooker and scout bees
        for _ in range(self.num_onlookers):
            self.onlooker_bees.append(ABCBee(role='Onlooker'))
        
        for _ in range(self.num_scouts):
            self.scout_bees.append(ABCBee(role='Scout'))
    
    def _evaluate_food_source(self, weights: np.ndarray, pso_proposal: float, 
                             aco_proposal: float, context: str) -> float:
        """Evaluate the quality of a weight combination."""
        lambda_pso, lambda_aco = weights
        
        # Simulate the effect of different weight combinations
        combined_score = lambda_pso * pso_proposal + lambda_aco * aco_proposal
        
        # Context-specific bonuses
        context_bonus = 0.0
        if context == "multilingual" and lambda_pso > 0.6:
            context_bonus = 0.1  # Favor PSO for multilingual tasks
        elif context == "specialized" and lambda_aco > 0.6:
            context_bonus = 0.1  # Favor ACO for specialized tasks
        
        # Penalty for extreme weights (encourage balance)
        balance_penalty = 0.1 * max(0, abs(lambda_pso - lambda_aco) - 0.5)
        
        return combined_score + context_bonus - balance_penalty
    
    def optimize_weights(self, pso_proposal: float, aco_proposal: float, 
                        context: str) -> Tuple[float, float]:
        """Meta-optimization to find optimal mixing weights."""
        logger.info("ABC: Optimizing strategic mixing weights...")
        
        # Check if conflict resolution is needed
        conflict_score = abs(pso_proposal - aco_proposal)
        
        if conflict_score <= self.conflict_threshold:
            logger.info(f"  - Low conflict ({conflict_score:.3f}), using balanced weights.")
            return (0.5, 0.5)
        
        logger.info(f"  - High conflict detected ({conflict_score:.3f}), optimizing...")
        
        # Employed bee phase: exploit current food sources
        for bee in self.employed_bees:
            # Generate neighbor solution
            neighbor = bee.food_source + np.random.normal(0, 0.1, 2)
            neighbor = np.abs(neighbor)
            neighbor = neighbor / np.sum(neighbor)  # Normalize
            
            # Evaluate neighbor
            current_fitness = self._evaluate_food_source(bee.food_source, pso_proposal, aco_proposal, context)
            neighbor_fitness = self._evaluate_food_source(neighbor, pso_proposal, aco_proposal, context)
            
            # Greedy selection
            if neighbor_fitness > current_fitness:
                bee.food_source = neighbor
                bee.fitness = neighbor_fitness
                bee.trial_count = 0
            else:
                bee.trial_count += 1
        
        # Onlooker bee phase: probabilistic selection
        total_fitness = sum(bee.fitness for bee in self.employed_bees if bee.fitness > 0)
        
        for onlooker in self.onlooker_bees:
            if total_fitness > 0:
                # Roulette wheel selection
                probabilities = [bee.fitness / total_fitness for bee in self.employed_bees]
                selected_idx = np.random.choice(len(self.employed_bees), p=probabilities)
                selected_bee = self.employed_bees[selected_idx]
                
                # Follow selected employed bee
                onlooker.food_source = selected_bee.food_source.copy()
                onlooker.fitness = selected_bee.fitness
        
        # Scout bee phase: abandon poor solutions and explore
        for bee in self.employed_bees:
            if bee.trial_count > self.limit:
                # Scout explores new random solution
                bee.food_source = np.random.rand(2)
                bee.food_source = bee.food_source / np.sum(bee.food_source)
                bee.fitness = self._evaluate_food_source(bee.food_source, pso_proposal, aco_proposal, context)
                bee.trial_count = 0
                logger.info("  - Scout bee exploring new solution space.")
        
        # Return best found weights
        best_bee = max(self.employed_bees, key=lambda b: b.fitness)
        weights = best_bee.food_source
        
        logger.info(f"  - Optimal weights found: λ_PSO={weights[0]:.2f}, λ_ACO={weights[1]:.2f}")
        return tuple(weights)
    
    def allocate_agent_roles(self, agent_performance: Dict[str, float], 
                           num_active_tasks: int) -> Dict[str, str]:
        """Allocate agents to ABC roles based on performance."""
        if not agent_performance:
            return {}
        
        agents = list(agent_performance.keys())
        performances = [agent_performance[agent] for agent in agents]
        sorted_indices = np.argsort(performances)[::-1]  # Sort descending
        
        # Adaptive role allocation based on task load
        employed_ratio = min(0.6, num_active_tasks / len(agents))
        scout_ratio = 0.2
        onlooker_ratio = 1.0 - employed_ratio - scout_ratio
        
        num_employed = max(1, int(len(agents) * employed_ratio))
        num_scouts = max(1, int(len(agents) * scout_ratio))
        
        role_assignments = {}
        for i, idx in enumerate(sorted_indices):
            agent = agents[idx]
            if i < num_employed:
                role_assignments[agent] = 'Employed'
            elif i >= len(agents) - num_scouts:
                role_assignments[agent] = 'Scout'
            else:
                role_assignments[agent] = 'Onlooker'
        
        logger.info(f"ABC: Allocated {num_employed} Employed, {len(agents)-num_employed-num_scouts} Onlookers, {num_scouts} Scouts")
        return role_assignments

class EnhancedBioOptimizer:
    """Enhanced bio-inspired optimizer with complete ABC/PSO/ACO implementation."""
    
    def __init__(self, pso_params: Optional[Dict] = None, aco_params: Optional[Dict] = None,
                 abc_params: Optional[Dict] = None, seed: Optional[int] = None):
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize optimizers with enhanced parameters
        pso_config = pso_params or {"dim": 128, "num_particles": 20, "w": 0.7, "c1": 1.5, "c2": 1.5}
        aco_config = aco_params or {"evaporation_rate": 0.5, "alpha": 1.0, "beta": 2.0}
        abc_config = abc_params or {"num_employed": 10, "num_onlookers": 10, "num_scouts": 5, "conflict_threshold": 0.05}
        
        self.pso = PSOOptimizer(**pso_config)
        self.aco = ACOOptimizer(**aco_config)
        self.abc = ABCOptimizer(**abc_config)
        self.safety_constraints = SafetyConstraints()
        
        logger.info("Enhanced BioOptimizer initialized with complete ABC/PSO/ACO algorithms.")
    
    def run_optimization_cycle(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one complete bio-optimization cycle with all algorithms."""
        logger.info("=== Starting Enhanced Bio-Optimization Cycle ===")
        
        # Extract system state
        agent_fitness = system_state.get("agent_fitness", {})
        successful_paths = system_state.get("successful_paths", {})
        context = system_state.get("context", "default")
        num_active_tasks = system_state.get("num_active_tasks", 1)
        
        # 1. PSO Tactical Optimization (Mesoscopic)
        pso_result = self.pso.update(agent_fitness)
        pso_proposal_strength = pso_result.get("global_best_fitness", 0.0)
        
        # 2. ACO Memory Update and Historical Analysis (Microscopic)
        pheromone_map = self.aco.update_pheromones(successful_paths)
        best_historical_path, aco_proposal_strength = self.aco.get_best_historical_path()
        
        # 3. ABC Strategic Meta-Optimization (Macroscopic)
        optimal_weights = self.abc.optimize_weights(pso_proposal_strength, aco_proposal_strength, context)
        role_assignments = self.abc.allocate_agent_roles(agent_fitness, num_active_tasks)
        
        # 4. Safety Constraint Validation
        edge_weights = {f"edge_{i}": w for i, w in enumerate(pso_result["pso_global_best"][:10])}
        safety_valid = self.safety_constraints.validate_safety_threshold(edge_weights)
        
        if not safety_valid:
            logger.warning("Safety constraints violated, applying conservative fallback.")
            optimal_weights = (0.5, 0.5)  # Conservative fallback
        
        # 5. Compile Results
        optimization_result = {
            "pso_global_best": pso_result["pso_global_best"],
            "pheromone_levels": pheromone_map,
            "conflict_weights": optimal_weights,
            "role_assignments": role_assignments,
            "best_historical_path": best_historical_path,
            "convergence_measure": pso_result.get("convergence_measure", 1.0),
            "safety_validated": safety_valid,
            "cycle_timestamp": np.random.randint(1000000)  # Simulation timestamp
        }
        
        logger.info("=== Bio-Optimization Cycle Complete ===")
        return optimization_result

# Enhanced Demo
if __name__ == "__main__":
    print("=" * 80)
    print("DEMO: Enhanced Hierarchical Bio-Optimization (ABC -> PSO -> ACO)")
    print("=" * 80)
    
    optimizer = EnhancedBioOptimizer(seed=42)
    
    # Enhanced scenario with more realistic data
    enhanced_state = {
        "context": "multilingual",
        "num_active_tasks": 3,
        "agent_fitness": {
            "Agent_A": 0.78, "Agent_B": 0.65, "Agent_C": 0.82, 
            "Agent_D": 0.71, "Agent_E": 0.59
        },
        "successful_paths": {
            "(task1, Agent_A)": 0.85, "(task1, Agent_C)": 0.91,
            "(task2, Agent_B)": 0.67, "(task3, Agent_A)": 0.74
        }
    }
    
    print("\n[Enhanced Optimization Cycle]\n")
    result = optimizer.run_optimization_cycle(enhanced_state)
    
    print("\n==> Complete Optimization Result:")
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value[:5]}... (showing first 5 elements)")
        elif isinstance(value, dict) and len(value) > 5:
            print(f"  {key}: {dict(list(value.items())[:3])}... (showing first 3)")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)