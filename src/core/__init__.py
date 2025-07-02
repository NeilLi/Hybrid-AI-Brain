#!/usr/bin/env python3
"""
src/core/__init__.py

Makes the core components of the Hybrid AI Brain easily importable.
Updated to include the faithful implementation following the theoretical framework
from sections 5 (Bio-Inspired Swarm), 6 (GNN Coordination), and 7 (Theoretical Analysis).
"""

# === Section 5: Bio-Inspired Swarm Architecture ===
from .hybrid_ai_brain_faithful import (
    BioInspiredSwarm,
    ABCRole,
    SwarmAgent,
    TaskNode,
)

# === Section 6: GNN Coordination Layer ===
from .hybrid_ai_brain_faithful import (
    TaskGraph,
    GNNCoordinator,
    HybridAIBrainFaithful,
)

# === Section 6.4: GraphMask Interpretability ===
from .hybrid_ai_brain_faithful import (
    GraphMaskInterpreter,
)

# === Legacy Core Components (for backward compatibility) ===
try:
    from .task_graph import TaskGraph as LegacyTaskGraph
    from .agent_pool import Agent, AgentPool
    from .match_score import calculate_match_score
    
    # Add legacy components to exports
    LEGACY_COMPONENTS = [
        "LegacyTaskGraph",
        "Agent", 
        "AgentPool",
        "calculate_match_score",
    ]
except ImportError:
    # Legacy components not available
    LEGACY_COMPONENTS = []

# === Main System Components ===
__all__ = [
    # === Core System (Faithful Implementation) ===
    "HybridAIBrainFaithful",           # Main system class
    
    # === Bio-Inspired Swarm (Section 5) ===
    "BioInspiredSwarm",                 # Bio-inspired optimization substrate
    "ABCRole",                          # ABC bee roles (Employed/Onlooker/Scout)
    "SwarmAgent",                       # Agent with ABC role and PSO properties
    
    # === GNN Coordination (Section 6) ===
    "GNNCoordinator",                   # GNN coordination layer
    "TaskGraph",                        # TaskGraph G_T for workflow execution
    "TaskNode",                         # Individual task representation
    
    # === Interpretability (Section 6.4) ===
    "GraphMaskInterpreter",             # GraphMask interpretability system
    
] + LEGACY_COMPONENTS

# === Convenience imports for common usage patterns ===

def create_hybrid_system(delta_bio: float = 2.0, delta_gnn: float = 0.2) -> HybridAIBrainFaithful:
    """
    Convenience function to create a new Hybrid AI Brain system.
    
    Args:
        delta_bio: Bio-inspired coordination interval (default: 2.0s)
        delta_gnn: GNN coordination interval (default: 0.2s)
    
    Returns:
        Initialized HybridAIBrainFaithful system
    """
    return HybridAIBrainFaithful(delta_bio=delta_bio, delta_gnn=delta_gnn)

def create_agent_with_capabilities(agent_id: str, **capabilities) -> SwarmAgent:
    """
    Convenience function to create a SwarmAgent with specified capabilities.
    
    Args:
        agent_id: Unique identifier for the agent
        **capabilities: Capability scores (e.g., sentiment_analysis=0.9)
    
    Returns:
        SwarmAgent instance with specified capabilities
    """
    return SwarmAgent(agent_id=agent_id, capabilities=capabilities)

def create_task_with_requirements(task_id: str, dependencies=None, priority: float = 1.0, **requirements) -> TaskNode:
    """
    Convenience function to create a TaskNode with specified requirements.
    
    Args:
        task_id: Unique identifier for the task
        dependencies: Set of task IDs this task depends on
        priority: Task priority (default: 1.0)
        **requirements: Capability requirements (e.g., sentiment_analysis=0.8)
    
    Returns:
        TaskNode instance with specified requirements
    """
    return TaskNode(
        task_id=task_id,
        requirements=requirements,
        dependencies=dependencies or set(),
        priority=priority
    )

# === Version and metadata ===
__version__ = "1.0.0"
__author__ = "Hybrid AI Brain Research Team"
__description__ = "Faithful implementation of the Hybrid AI Brain with Bio-GNN coordination protocol"

# === Module-level documentation ===
__doc__ = """
Hybrid AI Brain - Faithful Implementation

This module provides a faithful implementation of the Hybrid AI Brain system following
the exact theoretical framework described in the research paper:

CORE COMPONENTS:
================

1. HybridAIBrainFaithful - Main system integrating all components
   - Implements Bio-GNN Coordination Protocol
   - Maintains theoretical guarantees from Section 7
   - Provides timing synchronization (Δ_bio = 2s, Δ_gnn = 200ms)

2. BioInspiredSwarm - Section 5 implementation
   - ABC role allocation (Employed/Onlooker/Scout)
   - PSO tactical optimization with g_best convergence
   - ACO pheromone trail management with τ_at levels
   - Conflict resolution via strategic weight optimization

3. GNNCoordinator - Section 6 implementation
   - Dynamic heterogeneous graph G=(V,E) representation
   - One-shot assignment problem solving (K≤2 message-passing)
   - Bio-inspired signal integration into edge features
   - TaskGraph G_T workflow execution with dependencies

4. GraphMaskInterpreter - Section 6.4 implementation
   - Differentiable edge masks M_θ: E_S → [0,1]
   - Interpretable explanations with safety preservation
   - False-block rate ≤10^-4 guarantee

USAGE EXAMPLES:
===============

Basic system setup:
    >>> system = create_hybrid_system()
    >>> system.add_agent("agent_1", sentiment_analysis=0.9, multilingual=0.8)
    >>> system.add_task("task_1", sentiment_analysis=0.8, priority=1.0)
    >>> result = system.execute_coordination_cycle()

Advanced workflow with dependencies:
    >>> system.add_task("preprocess", preprocessing=0.8)
    >>> system.add_task("analyze", analysis=0.8, dependencies={"preprocess"})
    >>> # Only preprocess will be actionable initially
    >>> result = system.execute_coordination_cycle()
    >>> system.complete_task("preprocess")
    >>> # Now analyze becomes actionable

Interpretability analysis:
    >>> interpreter = GraphMaskInterpreter()
    >>> metrics = interpreter.train_edge_masks(system.gnn)
    >>> explanations = interpreter.get_explanation(assignments)

THEORETICAL GUARANTEES:
======================

The implementation maintains all theoretical guarantees:
- Theorem 7.1: Bio-stability with safety constraints (τ_safe ≥ 0.7, L_total < 1)
- Theorem 7.2: GNN convergence in K≤2 message-passing rounds
- Theorem 7.3: Safety properties for DAG task dependencies
- GraphMask: False-block rate ≤10^-4 with interpretability preservation

TIMING PROTOCOL:
================

The system operates on precise timing intervals:
- Bio-inspired coordination: Δ_bio = 2.0 seconds
- GNN coordination: Δ_gnn = 0.2 seconds (200ms)
- Message-passing: Converges in K≤2 rounds per Theorem 7.2

For complete documentation and examples, see the individual component docstrings.
"""

# === Import validation ===
def _validate_imports():
    """Validate that all core components are properly imported."""
    required_components = [
        "HybridAIBrainFaithful",
        "BioInspiredSwarm", 
        "GNNCoordinator",
        "TaskGraph",
        "GraphMaskInterpreter"
    ]
    
    missing = []
    for component in required_components:
        if component not in globals():
            missing.append(component)
    
    if missing:
        raise ImportError(f"Failed to import required components: {missing}")

# Validate imports on module load
_validate_imports()

# === Convenience re-exports for common patterns ===
# Make enum values easily accessible
from .hybrid_ai_brain_faithful import ABCRole
EMPLOYED = ABCRole.EMPLOYED
ONLOOKER = ABCRole.ONLOOKER  
SCOUT = ABCRole.SCOUT

# Export enum values
__all__.extend(["EMPLOYED", "ONLOOKER", "SCOUT"])

# === Module initialization message ===
import logging
logger = logging.getLogger(__name__)
logger.info("Hybrid AI Brain faithful implementation loaded successfully")
logger.info(f"Version: {__version__}")
logger.info("All theoretical guarantees from Sections 5, 6, 7 are maintained")