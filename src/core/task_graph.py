#!/usr/bin/env python3
"""
src/core/task_graph.py

Defines the TaskGraph class, which represents a complex task as a directed acyclic
graph (DAG) with rich node and edge attributes. This class provides methods for
dynamic subtask creation, explicit dependency management, and topological sorting
to ensure valid, deadlock-free execution ordering. The TaskGraph is foundational
for multi-agent coordination and reasoning workflows in the Hybrid AI Brain.
"""


import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger("hybrid_ai_brain.task_graph")

@dataclass
class TaskGraph:
    """Models a task as a directed acyclic graph (DAG) with node and edge attributes."""
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_subtask(
        self,
        task_id: str,
        required_capabilities: np.ndarray,
        **kwargs: Any
    ) -> None:
        norm = np.linalg.norm(required_capabilities)
        if norm > 0:
            normalized_caps = required_capabilities / norm
        else:
            raise ValueError("required_capabilities vector cannot be all zeros.")
        self.graph.add_node(task_id, required_capabilities=normalized_caps, **kwargs)
        logger.info(f"Added subtask '{task_id}'.")

    def add_dependency(
        self,
        from_task: str,
        to_task: str,
        cost: float = 0.0,
        risk: float = 0.0
    ) -> None:
        if from_task not in self.graph or to_task not in self.graph:
            raise ValueError("Both tasks must exist before adding a dependency.")
        self.graph.add_edge(from_task, to_task, cost=cost, risk=risk)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_task, to_task)
            raise ValueError(f"Dependency '{from_task}->{to_task}' would create a cycle.")
        logger.info(f"Added dependency '{from_task}' -> '{to_task}' with risk={risk}.")

    def get_subtask(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self.graph:
            raise KeyError(f"Task '{task_id}' not found.")
        return dict(self.graph.nodes[task_id])

    def get_all_subtasks(self) -> List[str]:
        return list(self.graph.nodes)

    def get_dependencies(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges)

    def get_edge_attributes(self, from_task: str, to_task: str) -> Dict[str, Any]:
        if not self.graph.has_edge(from_task, to_task):
            raise KeyError(f"No dependency from '{from_task}' to '{to_task}'.")
        return dict(self.graph.edges[from_task, to_task])

    def topological_order(self) -> List[str]:
        """Returns a list of subtasks in topological order (for valid execution)."""
        return list(nx.topological_sort(self.graph))

    def add_subtasks_bulk(self, tasks: Dict[str, np.ndarray]) -> None:
        """Add multiple subtasks at once. tasks: dict of task_id -> required_capabilities"""
        for task_id, caps in tasks.items():
            self.add_subtask(task_id, caps)

    def add_dependencies_bulk(self, dependencies: List[Tuple[str, str, float, float]]) -> None:
        """
        Add multiple dependencies at once.
        Each tuple: (from_task, to_task, cost, risk)
        """
        for from_task, to_task, cost, risk in dependencies:
            self.add_dependency(from_task, to_task, cost, risk)

    def to_dict(self) -> Dict[str, Any]:
        """Export the graph as a dict for serialization/testing."""
        return {
            "nodes": [
                {"id": n, **self.graph.nodes[n]} for n in self.graph.nodes
            ],
            "edges": [
                {"from": u, "to": v, **self.graph.edges[u, v]}
                for u, v in self.graph.edges
            ],
        }
    def topological_sort(self):
        return self.topological_order()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraph":
        tg = cls()
        for node in d.get("nodes", []):
            id = node.pop("id")
            tg.graph.add_node(id, **node)
        for edge in d.get("edges", []):
            from_task = edge.pop("from")
            to_task = edge.pop("to")
            tg.graph.add_edge(from_task, to_task, **edge)
        return tg

    def __repr__(self) -> str:
        return (f"TaskGraph(nodes={list(self.graph.nodes)}, "
                f"dependencies={list(self.graph.edges)})")
