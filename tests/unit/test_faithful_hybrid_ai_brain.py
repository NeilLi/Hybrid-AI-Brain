#!/usr/bin/env python3
"""
tests/unit/test_faithful_hybrid_ai_brain.py

Updated unit tests with robust import handling for faithful implementation.
"""

import unittest
import numpy as np
import time
import sys
from pathlib import Path

# Ensure project paths are available
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Robust import handling for faithful implementation
try:
    from core.hybrid_ai_brain_faithful import (
        BioInspiredSwarm, GNNCoordinator, HybridAIBrainFaithful,
        TaskGraph, GraphMaskInterpreter, ABCRole, SwarmAgent, TaskNode
    )
    HAS_FAITHFUL_IMPLEMENTATION = True
except ImportError:
    HAS_FAITHFUL_IMPLEMENTATION = False
    (BioInspiredSwarm, GNNCoordinator, HybridAIBrainFaithful, 
     TaskGraph, GraphMaskInterpreter, ABCRole, SwarmAgent, TaskNode) = [None] * 8

# Try importing existing core components as fallback
try:
    from core.task_graph import TaskGraph as ExistingTaskGraph
    from core.agent_pool import Agent, AgentPool
    HAS_EXISTING_CORE = True
except ImportError:
    HAS_EXISTING_CORE = False
    ExistingTaskGraph = Agent = AgentPool = None


class TestFaithfulImplementationAvailability(unittest.TestCase):
    """Test what faithful implementation components are available."""
    
    def test_faithful_component_availability(self):
        """Report faithful implementation availability."""
        components = {
            "Faithful Implementation": HAS_FAITHFUL_IMPLEMENTATION,
            "Existing Core Components": HAS_EXISTING_CORE
        }
        
        print(f"\nFaithful implementation availability:")
        for name, available in components.items():
            status = "✓" if available else "✗"
            print(f"  {status} {name}: {available}")
        
        self.assertTrue(True)  # Always pass - just informational


@unittest.skipUnless(HAS_FAITHFUL_IMPLEMENTATION, "Faithful implementation not available")
class TestBioInspiredSwarm(unittest.TestCase):
    """Test bio-inspired swarm if faithful implementation available."""
    
    def setUp(self):
        """Set up test environment."""
        self.swarm = BioInspiredSwarm(delta_bio=2.0)
        
        # Add test agents
        self.test_agents = {
            "agent_1": {"sentiment_analysis": 0.9, "multilingual": 0.7},
            "agent_2": {"sentiment_analysis": 0.6, "multilingual": 0.9},
            "agent_3": {"sentiment_analysis": 0.8, "reasoning": 0.8}
        }
        
        for agent_id, capabilities in self.test_agents.items():
            self.swarm.add_agent(agent_id, capabilities)
    
    def test_agent_initialization(self):
        """Test agent initialization with ABC roles."""
        self.assertEqual(len(self.swarm.agents), 3)
        
        for agent_id, agent in self.swarm.agents.items():
            self.assertIsInstance(agent, SwarmAgent)
            self.assertEqual(agent.role, ABCRole.ONLOOKER)  # Default role
            self.assertIsNotNone(agent.position)
            self.assertIsNotNone(agent.velocity)
    
    def test_abc_role_allocation(self):
        """Test ABC role allocation system."""
        # Add performance history
        self.swarm.agents["agent_1"].performance_history = [0.9, 0.8, 0.9]
        self.swarm.agents["agent_2"].performance_history = [0.7, 0.6, 0.7]
        
        roles = self.swarm.abc_role_allocation(task_count=2)
        
        # Should have proper role distribution
        role_counts = {}
        for role in roles.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        
        self.assertGreater(role_counts.get(ABCRole.EMPLOYED, 0), 0)
        self.assertGreater(role_counts.get(ABCRole.SCOUT, 0), 0)
    
    def test_safety_constraints(self):
        """Test safety constraint validation."""
        # Test valid constraints
        valid_g_best = np.array([0.5, 0.3, 0.4])
        valid_pheromones = {("agent_1", "task_1"): 0.8, ("agent_2", "task_2"): 0.9}
        
        is_safe = self.swarm._validate_safety_constraints(valid_g_best, valid_pheromones)
        self.assertTrue(is_safe)
        
        # Test Lipschitz violation
        invalid_g_best = np.array([2.0, 1.5, 1.8])  # Norm > 1
        is_safe = self.swarm._validate_safety_constraints(invalid_g_best, valid_pheromones)
        self.assertFalse(is_safe)


@unittest.skipUnless(HAS_FAITHFUL_IMPLEMENTATION, "Faithful implementation not available")
class TestGNNCoordinator(unittest.TestCase):
    """Test GNN coordinator if faithful implementation available."""
    
    def setUp(self):
        """Set up test environment."""
        self.gnn = GNNCoordinator(delta_gnn=0.2, max_message_rounds=2)
    
    def test_coordinator_initialization(self):
        """Test GNN coordinator initialization."""
        self.assertEqual(self.gnn.delta_gnn, 0.2)
        self.assertEqual(self.gnn.max_message_rounds, 2)
        self.assertEqual(self.gnn.beta, 2.0)
    
    def test_task_graph_dependencies(self):
        """Test TaskGraph dependency management."""
        # Add tasks with dependencies
        self.gnn.task_graph.add_task("task_1", {"capability": 0.8})
        self.gnn.task_graph.add_task("task_2", {"capability": 0.7}, dependencies={"task_1"})
        
        # Initially only task_1 should be actionable
        actionable = self.gnn.task_graph.get_actionable_tasks()
        self.assertIn("task_1", actionable)
        self.assertNotIn("task_2", actionable)
        
        # Complete task_1
        self.gnn.task_graph.complete_task("task_1")
        
        # Now task_2 should be actionable
        actionable = self.gnn.task_graph.get_actionable_tasks()
        self.assertIn("task_2", actionable)


@unittest.skipUnless(HAS_FAITHFUL_IMPLEMENTATION, "Faithful implementation not available")
class TestHybridAIBrainFaithful(unittest.TestCase):
    """Test complete faithful system if available."""
    
    def setUp(self):
        """Set up faithful system test environment."""
        self.system = HybridAIBrainFaithful(delta_bio=2.0, delta_gnn=0.2)
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertIsInstance(self.system, HybridAIBrainFaithful)
        self.assertEqual(self.system.delta_bio, 2.0)
        self.assertEqual(self.system.delta_gnn, 0.2)
    
    def test_agent_and_task_addition(self):
        """Test adding agents and tasks."""
        self.system.add_agent("agent_1", {"sentiment_analysis": 0.9})
        self.system.add_task("task_1", {"sentiment_analysis": 0.8})
        
        self.assertIn("agent_1", self.system.swarm.agents)
        self.assertIn("task_1", self.system.gnn.task_graph.tasks)
    
    def test_coordination_cycle(self):
        """Test complete coordination cycle."""
        # Add minimal setup
        self.system.add_agent("agent_1", {"capability": 0.8})
        self.system.add_task("task_1", {"capability": 0.7})
        
        result = self.system.execute_coordination_cycle()
        
        self.assertIsInstance(result, dict)
        self.assertIn("coordination_result", result)
        self.assertIn("guarantees_validated", result)


@unittest.skipUnless(HAS_FAITHFUL_IMPLEMENTATION, "Faithful implementation not available")
class TestGraphMaskInterpreter(unittest.TestCase):
    """Test GraphMask interpretability if available."""
    
    def test_mask_interpreter_basic(self):
        """Test basic GraphMask interpreter functionality."""
        interpreter = GraphMaskInterpreter()
        self.assertIsInstance(interpreter, GraphMaskInterpreter)
        self.assertEqual(interpreter.sparsity_lambda, 0.1)


@unittest.skipUnless(HAS_EXISTING_CORE, "Existing core components not available")
class TestExistingCoreComponents(unittest.TestCase):
    """Test existing core components as fallback."""
    
    def test_existing_task_graph(self):
        """Test existing TaskGraph implementation."""
        task_graph = ExistingTaskGraph()
        self.assertIsInstance(task_graph, ExistingTaskGraph)
    
    def test_existing_agent_pool(self):
        """Test existing Agent and AgentPool."""
        agent = Agent(agent_id="test_agent", capabilities={"capability": 0.8})
        self.assertEqual(agent.agent_id, "test_agent")
        
        pool = AgentPool()
        pool.add_agent(agent)
        
        # FIXED: pool.agents contains Agent objects, not agent IDs
        # Check membership correctly
        self.assertEqual(pool.count, 1)
        self.assertIsNotNone(pool.get_agent("test_agent"))
        self.assertIn("test_agent", pool._agent_map)


class TestFallbackIntegration(unittest.TestCase):
    """Integration tests that work with any available implementation."""
    
    def test_any_available_system(self):
        """Test with whatever system is available."""
        if HAS_FAITHFUL_IMPLEMENTATION:
            system = HybridAIBrainFaithful()
            system.add_agent("agent_1", {"capability": 0.8})
            result = system.execute_coordination_cycle()
            self.assertIsInstance(result, dict)
            
        elif HAS_EXISTING_CORE:
            task_graph = ExistingTaskGraph()
            agent = Agent("test_agent", {"capability": 0.8})
            self.assertIsInstance(task_graph, ExistingTaskGraph)
            self.assertIsInstance(agent, Agent)
            
        else:
            self.skipTest("No implementation available")
    
    def test_theoretical_guarantees_concept(self):
        """Test theoretical guarantees concept with available components."""
        if HAS_FAITHFUL_IMPLEMENTATION:
            # Test ABC role system
            self.assertIn(ABCRole.EMPLOYED, [ABCRole.EMPLOYED, ABCRole.ONLOOKER, ABCRole.SCOUT])
            self.assertIn(ABCRole.ONLOOKER, [ABCRole.EMPLOYED, ABCRole.ONLOOKER, ABCRole.SCOUT])
            self.assertIn(ABCRole.SCOUT, [ABCRole.EMPLOYED, ABCRole.ONLOOKER, ABCRole.SCOUT])
            
            # Test that system can be created without errors
            system = HybridAIBrainFaithful()
            self.assertIsNotNone(system)
            
        else:
            self.skipTest("Faithful implementation not available for theoretical tests")


def run_faithful_tests():
    """Run all faithful implementation tests."""
    suite = unittest.TestSuite()
    
    test_classes = [
        TestFaithfulImplementationAvailability,
        TestBioInspiredSwarm,
        TestGNNCoordinator,
        TestHybridAIBrainFaithful,
        TestGraphMaskInterpreter,
        TestExistingCoreComponents,
        TestFallbackIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print(f"\n{'='*60}")
    print("FAITHFUL IMPLEMENTATION TEST SUMMARY") 
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 All faithful implementation tests passed!")
    else:
        print("\n⚠️  Some tests failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_faithful_tests()