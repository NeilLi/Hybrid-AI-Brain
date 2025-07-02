#!/usr/bin/env python3
"""
src/governance/manifest_governance.py

Implements the declarative governance layer with manifest-driven control
that operates over the bio-inspired coordination system, as described in
the "Hybrid AI Brain" paper Section on Declarative Governance.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from abc import ABC, abstractmethod

# Import coordination components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'coordination'))

logger = logging.getLogger("hybrid_ai_brain.manifest_governance")

class OperationalDomain(Enum):
    """Operational domains as described in the paper."""
    PRECISION = "precision"     # High accuracy, conservative exploration
    ADAPTIVE = "adaptive"       # Balanced exploration and exploitation
    EXPLORATION = "exploration" # High exploration, rapid adaptation

class ManifestPhase(Enum):
    """Phases within a manifest execution cycle."""
    STRATEGY = "strategy"       # ABC sets strategic weights
    EXPLORATION = "exploration" # ACO/ABC recruitment and team formation
    TACTICAL = "tactical"       # PSO tactical optimization
    ORCHESTRATION = "orchestration"  # GNN coordination
    FEEDBACK = "feedback"       # Performance feedback and learning

@dataclass
class ManifestControl:
    """Manifest control parameters C_M and governance flag g_M."""
    exploration_weight: float = 0.5    # Balance between exploration/exploitation
    safety_enforcement: float = 0.8    # Safety constraint strictness
    coordination_aggressiveness: float = 0.6  # How aggressively to coordinate
    adaptation_rate: float = 0.3       # Rate of parameter adaptation
    governance_enabled: bool = True     # g_M flag from paper
    
    def to_vector(self) -> np.ndarray:
        """Convert control parameters to vector form C_M."""
        return np.array([
            self.exploration_weight,
            self.safety_enforcement,
            self.coordination_aggressiveness,
            self.adaptation_rate,
            float(self.governance_enabled)
        ])

@dataclass
class Manifest:
    """Represents a system manifest with control parameters and constraints."""
    manifest_id: str
    domain: OperationalDomain
    control: ManifestControl
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    active_until: Optional[float] = None
    
    def is_active(self, current_time: float) -> bool:
        """Check if manifest is currently active."""
        if self.active_until is not None:
            return current_time <= self.active_until
        return True
    
    def apply_domain_constraints(self) -> Dict[str, Any]:
        """Apply domain-specific constraints and parameters."""
        base_constraints = self.constraints.copy()
        
        if self.domain == OperationalDomain.PRECISION:
            base_constraints.update({
                "safety_threshold": 0.85,      # Higher safety requirement
                "exploration_limit": 0.2,      # Limited exploration
                "convergence_tolerance": 0.01,  # Strict convergence
                "lipschitz_bound": 0.7         # Conservative Lipschitz bound
            })
        elif self.domain == OperationalDomain.ADAPTIVE:
            base_constraints.update({
                "safety_threshold": 0.7,       # Standard safety
                "exploration_limit": 0.5,      # Balanced exploration
                "convergence_tolerance": 0.05,  # Moderate convergence
                "lipschitz_bound": 0.9         # Standard Lipschitz bound
            })
        elif self.domain == OperationalDomain.EXPLORATION:
            base_constraints.update({
                "safety_threshold": 0.6,       # Relaxed safety for exploration
                "exploration_limit": 0.8,      # High exploration
                "convergence_tolerance": 0.1,   # Relaxed convergence
                "lipschitz_bound": 0.95        # Aggressive Lipschitz bound
            })
        
        return base_constraints

class ManifestScheduler:
    """Determines execution phase based on domain and time."""
    
    def __init__(self):
        # Phase durations in seconds (can be domain-specific)
        self.phase_durations = {
            OperationalDomain.PRECISION: {
                ManifestPhase.STRATEGY: 0.3,
                ManifestPhase.EXPLORATION: 0.2,
                ManifestPhase.TACTICAL: 0.8,
                ManifestPhase.ORCHESTRATION: 0.6,
                ManifestPhase.FEEDBACK: 0.1
            },
            OperationalDomain.ADAPTIVE: {
                ManifestPhase.STRATEGY: 0.2,
                ManifestPhase.EXPLORATION: 0.4,
                ManifestPhase.TACTICAL: 0.6,
                ManifestPhase.ORCHESTRATION: 0.6,
                ManifestPhase.FEEDBACK: 0.2
            },
            OperationalDomain.EXPLORATION: {
                ManifestPhase.STRATEGY: 0.1,
                ManifestPhase.EXPLORATION: 0.6,
                ManifestPhase.TACTICAL: 0.4,
                ManifestPhase.ORCHESTRATION: 0.7,
                ManifestPhase.FEEDBACK: 0.2
            }
        }
    
    def get_current_phase(self, time_in_cycle: float, domain: OperationalDomain) -> ManifestPhase:
        """Determine current phase based on time and domain."""
        durations = self.phase_durations[domain]
        cumulative_time = 0.0
        
        for phase, duration in durations.items():
            cumulative_time += duration
            if time_in_cycle <= cumulative_time:
                return phase
        
        return ManifestPhase.FEEDBACK  # Default to feedback phase
    
    def get_cycle_duration(self, domain: OperationalDomain) -> float:
        """Get total cycle duration for a domain."""
        return sum(self.phase_durations[domain].values())

class ManifestService:
    """Service for managing and serving manifests."""
    
    def __init__(self):
        self.manifests: Dict[str, Manifest] = {}
        self.active_manifest: Optional[Manifest] = None
        self.manifest_history: List[Tuple[float, str]] = []
        self.service_available = True
        
    def register_manifest(self, manifest: Manifest):
        """Register a new manifest."""
        self.manifests[manifest.manifest_id] = manifest
        logger.info(f"Registered manifest {manifest.manifest_id} for domain {manifest.domain.value}")
    
    def activate_manifest(self, manifest_id: str, current_time: float) -> bool:
        """Activate a specific manifest."""
        if manifest_id not in self.manifests:
            logger.error(f"Manifest {manifest_id} not found")
            return False
        
        manifest = self.manifests[manifest_id]
        if not manifest.is_active(current_time):
            logger.warning(f"Manifest {manifest_id} is not active at time {current_time}")
            return False
        
        self.active_manifest = manifest
        self.manifest_history.append((current_time, manifest_id))
        logger.info(f"Activated manifest {manifest_id} for domain {manifest.domain.value}")
        return True
    
    def get_active_manifest(self, current_time: float) -> Optional[Manifest]:
        """Get currently active manifest."""
        if not self.service_available:
            logger.warning("Manifest service unavailable - graceful degradation")
            return self.active_manifest  # Return last known manifest
        
        if self.active_manifest and self.active_manifest.is_active(current_time):
            return self.active_manifest
        
        # Auto-select appropriate manifest if none active
        return self._auto_select_manifest(current_time)
    
    def _auto_select_manifest(self, current_time: float) -> Optional[Manifest]:
        """Automatically select appropriate manifest based on system state."""
        # Simple policy: cycle through domains
        cycle_time = current_time % 30.0  # 30-second cycles
        
        if cycle_time < 10.0:
            target_domain = OperationalDomain.PRECISION
        elif cycle_time < 20.0:
            target_domain = OperationalDomain.ADAPTIVE
        else:
            target_domain = OperationalDomain.EXPLORATION
        
        # Find manifest for target domain
        for manifest in self.manifests.values():
            if manifest.domain == target_domain and manifest.is_active(current_time):
                self.active_manifest = manifest
                return manifest
        
        return self.active_manifest  # Fallback to current

class GovernanceLayer:
    """Main governance layer implementing Algorithm 1 from the paper."""
    
    def __init__(self, bio_coordinator=None):
        self.manifest_service = ManifestService()
        self.scheduler = ManifestScheduler()
        self.bio_coordinator = bio_coordinator
        
        # Governance state
        self.current_state = {}
        self.manifest_stale_seconds = 0
        self.last_manifest_update = 0.0
        self.governance_history = []
        
        # Initialize default manifests
        self._initialize_default_manifests()
        
        logger.info("Governance layer initialized with manifest-driven control")
    
    def _initialize_default_manifests(self):
        """Initialize default manifests for each operational domain."""
        manifests = [
            Manifest(
                manifest_id="precision_manifest",
                domain=OperationalDomain.PRECISION,
                control=ManifestControl(
                    exploration_weight=0.2,
                    safety_enforcement=0.9,
                    coordination_aggressiveness=0.4,
                    adaptation_rate=0.1
                ),
                constraints={"max_concurrent_tasks": 3}
            ),
            Manifest(
                manifest_id="adaptive_manifest", 
                domain=OperationalDomain.ADAPTIVE,
                control=ManifestControl(
                    exploration_weight=0.5,
                    safety_enforcement=0.8,
                    coordination_aggressiveness=0.6,
                    adaptation_rate=0.3
                ),
                constraints={"max_concurrent_tasks": 5}
            ),
            Manifest(
                manifest_id="exploration_manifest",
                domain=OperationalDomain.EXPLORATION,
                control=ManifestControl(
                    exploration_weight=0.8,
                    safety_enforcement=0.6,
                    coordination_aggressiveness=0.8,
                    adaptation_rate=0.5
                ),
                constraints={"max_concurrent_tasks": 8}
            )
        ]
        
        for manifest in manifests:
            self.manifest_service.register_manifest(manifest)
        
        # Activate adaptive manifest by default
        self.manifest_service.activate_manifest("adaptive_manifest", time.time())
    
    def manifest_driven_coordination_cycle(self, current_state: Dict[str, Any], 
                                         current_time: float) -> Dict[str, Any]:
        """
        Algorithm 1 from paper: Manifest-Driven Coordination Cycle
        """
        logger.info(f"=== Manifest-Driven Governance Cycle (t={current_time:.1f}s) ===")
        
        # Step 1: Extract control parameters from active manifest
        active_manifest = self.manifest_service.get_active_manifest(current_time)
        
        if active_manifest is None:
            logger.warning("No active manifest - using emergency fallback")
            return self._emergency_fallback(current_state)
        
        # Update manifest staleness tracking
        if active_manifest != getattr(self, '_last_manifest', None):
            self.last_manifest_update = current_time
            self.manifest_stale_seconds = 0
            self._last_manifest = active_manifest
        else:
            self.manifest_stale_seconds = current_time - self.last_manifest_update
        
        # Step 2: Determine current phase based on domain
        cycle_duration = self.scheduler.get_cycle_duration(active_manifest.domain)
        time_in_cycle = current_time % cycle_duration
        current_phase = self.scheduler.get_current_phase(time_in_cycle, active_manifest.domain)
        
        logger.info(f"Domain: {active_manifest.domain.value}, Phase: {current_phase.value}")
        
        # Step 3: Extract control parameters (C_M, g_M)
        control_vector = active_manifest.control.to_vector()
        governance_enabled = active_manifest.control.governance_enabled
        domain_constraints = active_manifest.apply_domain_constraints()
        
        # Step 4: Execute bio-inspired optimization if governance enabled
        updated_state = current_state.copy()
        bio_result = None
        
        if governance_enabled and self.bio_coordinator is not None:
            # Adapt bio-optimization based on domain and phase
            bio_params = self._adapt_bio_parameters(active_manifest, current_phase, domain_constraints)
            
            # Update bio-coordinator parameters
            self._update_bio_coordinator_parameters(bio_params)
            
            # Execute bio-optimization
            bio_state = {
                "agent_fitness": current_state.get("agent_fitness", {}),
                "successful_paths": current_state.get("successful_paths", {}),
                "context": self._determine_context(active_manifest, current_phase),
                "num_active_tasks": current_state.get("num_active_tasks", 1),
                "domain_constraints": domain_constraints,
                "phase": current_phase.value
            }
            
            bio_result = self.bio_coordinator.run_optimization_cycle(bio_state)
            
            # Apply conflict resolution with domain-aware weights
            if bio_result:
                bio_result = self._apply_domain_conflict_resolution(bio_result, active_manifest, domain_constraints)
        
        # Step 5: Apply domain-aware state update function F(S_t, m_t, d_t)
        updated_state = self._apply_domain_update(current_state, control_vector, 
                                                active_manifest.domain, bio_result)
        
        # Step 6: Validate safety constraints and governance guarantees
        safety_validation = self._validate_governance_safety(updated_state, domain_constraints)
        
        # Step 7: Handle graceful degradation if manifest service unavailable
        if self.manifest_stale_seconds > 10.0:  # Service unavailable for >10s
            logger.warning(f"Manifest service stale for {self.manifest_stale_seconds:.1f}s - graceful degradation")
            updated_state = self._apply_graceful_degradation(updated_state)
        
        # Step 8: Record governance cycle for monitoring
        governance_record = {
            "timestamp": current_time,
            "manifest_id": active_manifest.manifest_id,
            "domain": active_manifest.domain.value,
            "phase": current_phase.value,
            "control_vector": control_vector.tolist(),
            "safety_validated": safety_validation,
            "bio_optimization_used": bio_result is not None,
            "manifest_stale_seconds": self.manifest_stale_seconds
        }
        self.governance_history.append(governance_record)
        
        logger.info(f"Governance cycle complete - safety validated: {safety_validation}")
        
        return {
            "updated_state": updated_state,
            "bio_result": bio_result,
            "manifest": active_manifest,
            "phase": current_phase,
            "safety_validated": safety_validation,
            "governance_record": governance_record
        }
    
    def _adapt_bio_parameters(self, manifest: Manifest, phase: ManifestPhase, 
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt bio-inspired algorithm parameters based on manifest and phase."""
        base_params = {
            "pso_params": {"w": 0.7, "c1": 1.5, "c2": 1.5},
            "aco_params": {"evaporation_rate": 0.5, "alpha": 1.0, "beta": 2.0},
            "abc_params": {"conflict_threshold": 0.05}
        }
        
        # Domain-specific adaptations
        if manifest.domain == OperationalDomain.PRECISION:
            base_params["pso_params"]["w"] = 0.5  # Lower inertia for stability
            base_params["aco_params"]["evaporation_rate"] = 0.3  # Preserve memory longer
            base_params["abc_params"]["conflict_threshold"] = 0.03  # More sensitive to conflicts
            
        elif manifest.domain == OperationalDomain.EXPLORATION:
            base_params["pso_params"]["w"] = 0.9  # Higher inertia for exploration
            base_params["aco_params"]["evaporation_rate"] = 0.7  # Faster forgetting
            base_params["abc_params"]["conflict_threshold"] = 0.1  # Less sensitive to conflicts
        
        # Phase-specific adaptations
        if phase == ManifestPhase.EXPLORATION:
            base_params["pso_params"]["c1"] = 2.0  # Higher cognitive component
            base_params["abc_params"]["num_scouts"] = 8  # More scouts
            
        elif phase == ManifestPhase.TACTICAL:
            base_params["pso_params"]["c2"] = 2.0  # Higher social component
            base_params["abc_params"]["num_employed"] = 15  # More employed bees
        
        # Apply manifest control parameters
        exploration_factor = manifest.control.exploration_weight
        base_params["pso_params"]["c1"] *= (1 + exploration_factor)
        base_params["aco_params"]["alpha"] *= (1 + exploration_factor)
        
        return base_params
    
    def _update_bio_coordinator_parameters(self, bio_params: Dict[str, Any]):
        """Update bio-coordinator with new parameters."""
        if self.bio_coordinator is None:
            return
        
        # Update PSO parameters
        if hasattr(self.bio_coordinator, 'pso'):
            pso_params = bio_params.get("pso_params", {})
            for param, value in pso_params.items():
                if hasattr(self.bio_coordinator.pso, param):
                    setattr(self.bio_coordinator.pso, param, value)
        
        # Update ACO parameters
        if hasattr(self.bio_coordinator, 'aco'):
            aco_params = bio_params.get("aco_params", {})
            for param, value in aco_params.items():
                if hasattr(self.bio_coordinator.aco, param):
                    setattr(self.bio_coordinator.aco, param, value)
        
        # Update ABC parameters
        if hasattr(self.bio_coordinator, 'abc'):
            abc_params = bio_params.get("abc_params", {})
            for param, value in abc_params.items():
                if hasattr(self.bio_coordinator.abc, param):
                    setattr(self.bio_coordinator.abc, param, value)
    
    def _determine_context(self, manifest: Manifest, phase: ManifestPhase) -> str:
        """Determine context string based on manifest and phase."""
        if manifest.domain == OperationalDomain.PRECISION:
            return "precision_focused"
        elif manifest.domain == OperationalDomain.EXPLORATION:
            if phase == ManifestPhase.EXPLORATION:
                return "high_exploration"
            else:
                return "exploration_mode"
        elif phase == ManifestPhase.TACTICAL:
            return "tactical_optimization"
        else:
            return "adaptive_balanced"
    
    def _apply_domain_conflict_resolution(self, bio_result: Dict[str, Any], 
                                        manifest: Manifest, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply domain-specific conflict resolution adjustments."""
        if "conflict_weights" not in bio_result:
            return bio_result
        
        lambda_pso, lambda_aco = bio_result["conflict_weights"]
        
        # Domain-specific weight adjustments
        if manifest.domain == OperationalDomain.PRECISION:
            # Favor ACO (historical knowledge) for precision
            lambda_aco = min(0.8, lambda_aco + 0.2)
            lambda_pso = 1.0 - lambda_aco
            
        elif manifest.domain == OperationalDomain.EXPLORATION:
            # Favor PSO (exploration) for exploration domain
            lambda_pso = min(0.8, lambda_pso + 0.2)
            lambda_aco = 1.0 - lambda_pso
        
        # Apply safety enforcement
        safety_factor = manifest.control.safety_enforcement
        if safety_factor > 0.8:
            # Conservative adjustment toward balanced weights
            target_balance = 0.5
            lambda_pso = lambda_pso * (1 - safety_factor * 0.3) + target_balance * (safety_factor * 0.3)
            lambda_aco = 1.0 - lambda_pso
        
        bio_result["conflict_weights"] = (lambda_pso, lambda_aco)
        bio_result["domain_adjusted"] = True
        
        return bio_result
    
    def _apply_domain_update(self, current_state: Dict[str, Any], control_vector: np.ndarray,
                           domain: OperationalDomain, bio_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply domain-aware state update function F(S_t, m_t, d_t)."""
        updated_state = current_state.copy()
        
        # Extract control parameters
        exploration_weight = control_vector[0]
        safety_enforcement = control_vector[1]
        coordination_aggressiveness = control_vector[2]
        adaptation_rate = control_vector[3]
        
        # Apply domain-specific updates
        if domain == OperationalDomain.PRECISION:
            # Emphasize stability and safety
            updated_state["stability_bonus"] = 0.2
            updated_state["exploration_penalty"] = 0.1
            
        elif domain == OperationalDomain.EXPLORATION:
            # Emphasize exploration and adaptation
            updated_state["exploration_bonus"] = 0.3
            updated_state["adaptation_bonus"] = 0.2
            
        elif domain == OperationalDomain.ADAPTIVE:
            # Balanced approach
            updated_state["balance_bonus"] = 0.1
        
        # Apply bio-optimization results if available
        if bio_result is not None:
            updated_state["bio_optimization"] = bio_result
            updated_state["coordination_quality"] = self._calculate_coordination_quality(bio_result)
        
        # Apply adaptation rate to parameter updates
        if "agent_fitness" in current_state and bio_result:
            old_fitness = current_state["agent_fitness"]
            new_fitness = {}
            
            for agent_id, old_score in old_fitness.items():
                # Gradual adaptation based on adaptation rate
                if bio_result.get("role_assignments", {}).get(agent_id) == "Employed":
                    boost = 0.1 * adaptation_rate
                else:
                    boost = 0.0
                
                new_fitness[agent_id] = old_score + boost
            
            updated_state["agent_fitness"] = new_fitness
        
        return updated_state
    
    def _calculate_coordination_quality(self, bio_result: Dict[str, Any]) -> float:
        """Calculate coordination quality from bio-optimization results."""
        quality_factors = []
        
        # PSO convergence quality
        if "convergence_measure" in bio_result:
            convergence = bio_result["convergence_measure"]
            # Lower convergence measure (variance) = higher quality
            quality_factors.append(max(0.0, 1.0 - convergence))
        
        # Conflict resolution effectiveness
        if "conflict_weights" in bio_result:
            lambda_pso, lambda_aco = bio_result["conflict_weights"]
            # Penalize extreme weights, favor balanced solutions
            balance_score = 1.0 - abs(lambda_pso - lambda_aco)
            quality_factors.append(balance_score)
        
        # Safety validation
        if "safety_validated" in bio_result:
            safety_score = 1.0 if bio_result["safety_validated"] else 0.0
            quality_factors.append(safety_score)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def _validate_governance_safety(self, state: Dict[str, Any], 
                                  constraints: Dict[str, Any]) -> bool:
        """Validate that governance maintains safety constraints."""
        # Check safety threshold
        safety_threshold = constraints.get("safety_threshold", 0.7)
        coordination_quality = state.get("coordination_quality", 0.0)
        
        if coordination_quality < safety_threshold:
            logger.warning(f"Coordination quality {coordination_quality:.3f} below threshold {safety_threshold}")
            return False
        
        # Check Lipschitz constraint if bio-optimization was used
        if "bio_optimization" in state:
            bio_result = state["bio_optimization"]
            if "pso_global_best" in bio_result:
                weights = bio_result["pso_global_best"]
                if len(weights) > 0:
                    spectral_norm = np.linalg.norm(weights[:10])  # Check first 10 elements
                    lipschitz_bound = constraints.get("lipschitz_bound", 1.0)
                    
                    if spectral_norm >= lipschitz_bound:
                        logger.warning(f"Lipschitz constraint violated: {spectral_norm:.3f} >= {lipschitz_bound}")
                        return False
        
        # Check maximum concurrent tasks
        max_tasks = constraints.get("max_concurrent_tasks", float('inf'))
        current_tasks = state.get("num_active_tasks", 0)
        
        if current_tasks > max_tasks:
            logger.warning(f"Too many concurrent tasks: {current_tasks} > {max_tasks}")
            return False
        
        return True
    
    def _apply_graceful_degradation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply graceful degradation when manifest service is unavailable."""
        logger.info("Applying graceful degradation strategy")
        
        # Use conservative parameters
        degraded_state = state.copy()
        degraded_state["degradation_mode"] = True
        degraded_state["safety_multiplier"] = 1.5  # Extra safety margin
        
        # Reduce exploration and adaptation
        if "exploration_bonus" in degraded_state:
            degraded_state["exploration_bonus"] *= 0.5
        
        if "adaptation_rate" in degraded_state:
            degraded_state["adaptation_rate"] *= 0.3
        
        return degraded_state
    
    def _emergency_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency fallback when no manifest is available."""
        logger.error("Emergency fallback activated - no manifest available")
        
        fallback_state = state.copy()
        fallback_state["emergency_mode"] = True
        fallback_state["coordination_quality"] = 0.5  # Conservative estimate
        
        return {
            "updated_state": fallback_state,
            "bio_result": None,
            "manifest": None,
            "phase": ManifestPhase.FEEDBACK,
            "safety_validated": False,
            "governance_record": {
                "timestamp": time.time(),
                "emergency_fallback": True
            }
        }
    
    def get_governance_statistics(self) -> Dict[str, Any]:
        """Get governance layer statistics and performance metrics."""
        if not self.governance_history:
            return {}
        
        recent_history = self.governance_history[-20:]  # Last 20 cycles
        
        # Calculate domain distribution
        domain_counts = {}
        for record in recent_history:
            domain = record.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Calculate safety validation rate
        safety_validations = [r.get("safety_validated", False) for r in recent_history]
        safety_rate = np.mean(safety_validations) if safety_validations else 0.0
        
        # Calculate bio-optimization usage
        bio_usage = [r.get("bio_optimization_used", False) for r in recent_history]
        bio_usage_rate = np.mean(bio_usage) if bio_usage else 0.0
        
        return {
            "total_governance_cycles": len(self.governance_history),
            "recent_domain_distribution": domain_counts,
            "safety_validation_rate": safety_rate,
            "bio_optimization_usage_rate": bio_usage_rate,
            "current_manifest_stale_seconds": self.manifest_stale_seconds,
            "total_manifests": len(self.manifest_service.manifests)
        }

# Demo and Integration Test
if __name__ == "__main__":
    print("=" * 80)
    print("DEMO: Manifest-Driven Governance with Bio-Inspired Coordination")
    print("=" * 80)
    
    # Initialize governance layer
    governance = GovernanceLayer()
    
    # Simulate system state
    initial_state = {
        "agent_fitness": {
            "Agent_A": 0.8, "Agent_B": 0.6, "Agent_C": 0.9
        },
        "successful_paths": {
            "(agent_task, Agent_A, Task_1)": 0.85,
            "(agent_task, Agent_C, Task_2)": 0.78
        },
        "num_active_tasks": 2
    }
    
    print("\n[Running Governance Cycles Across Domains]\n")
    
    # Test different operational domains
    domains_to_test = [
        ("precision_manifest", 5),
        ("adaptive_manifest", 5), 
        ("exploration_manifest", 5)
    ]
    
    current_time = 0.0
    for manifest_id, cycles in domains_to_test:
        print(f"--- Testing {manifest_id} ---")
        
        # Activate specific manifest
        governance.manifest_service.activate_manifest(manifest_id, current_time)
        
        for cycle in range(cycles):
            result = governance.manifest_driven_coordination_cycle(initial_state, current_time)
            
            print(f"  Cycle {cycle + 1}: Domain={result['manifest'].domain.value}, "
                  f"Phase={result['phase'].value}, Safety={result['safety_validated']}")
            
            # Update state for next cycle
            initial_state = result["updated_state"]
            current_time += 2.0  # Advance time
        
        print()
    
    # Test graceful degradation
    print("--- Testing Graceful Degradation ---")
    governance.manifest_service.service_available = False
    governance.manifest_stale_seconds = 15.0  # Simulate stale service
    
    result = governance.manifest_driven_coordination_cycle(initial_state, current_time)
    print(f"  Degradation mode: {result['updated_state'].get('degradation_mode', False)}")
    print(f"  Safety validated: {result['safety_validated']}")
    
    # Show final statistics
    print("\n==> Governance Statistics:")
    stats = governance.get_governance_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
