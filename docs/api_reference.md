# API Reference

## Overview

This document provides comprehensive API documentation for the Hybrid AI Brain framework. The API is organized into logical modules corresponding to the system's architectural layers: Governance, Coordination, Safety, Memory, and Core components.

## Table of Contents

- [Core Components](#core-components)
- [Coordination Layer](#coordination-layer)
- [Memory Layer](#memory-layer)
- [Safety Layer](#safety-layer)
- [Governance Layer](#governance-layer)
- [Utilities](#utilities)
- [Configuration](#configuration)
- [Examples](#examples)

## Core Components

### TaskGraph

Represents complex tasks as directed acyclic graphs with rich node and edge attributes.

```python
from src.core.task_graph import TaskGraph
```

#### Constructor

```python
TaskGraph()
```

Creates an empty task graph backed by NetworkX DiGraph.

#### Methods

##### `add_subtask(task_id: str, required_capabilities: np.ndarray, **kwargs) -> None`

Adds a subtask node to the graph.

**Parameters:**
- `task_id`: Unique identifier for the subtask
- `required_capabilities`: Normalized capability vector (automatically normalized)
- `**kwargs`: Additional node attributes (e.g., description, priority)

**Raises:**
- `ValueError`: If required_capabilities is zero vector

**Example:**
```python
tg = TaskGraph()
tg.add_subtask("analyze_data", 
               np.array([0.8, 0.6, 0.2]), 
               description="Analyze FIFA match data")
```

##### `add_dependency(from_task: str, to_task: str, cost: float = 0.0, risk: float = 0.0) -> None`

Adds a dependency edge between two subtasks.

**Parameters:**
- `from_task`: Source task ID (must exist)
- `to_task`: Target task ID (must exist)
- `cost`: Edge cost (default: 0.0)
- `risk`: Edge risk level (default: 0.0)

**Raises:**
- `ValueError`: If tasks don't exist or dependency would create cycle

**Example:**
```python
tg.add_dependency("fetch_data", "analyze_data", cost=1.0, risk=0.1)
```

##### `topological_order() -> List[str]`

Returns tasks in topological execution order.

**Returns:**
- List of task IDs in valid execution order

**Example:**
```python
execution_order = tg.topological_order()
# ['fetch_data', 'analyze_data', 'generate_report']
```

##### `get_subtask(task_id: str) -> Dict[str, Any]`

Retrieves subtask attributes.

**Parameters:**
- `task_id`: Task identifier

**Returns:**
- Dictionary of task attributes including required_capabilities

**Raises:**
- `KeyError`: If task doesn't exist

##### `to_dict() -> Dict[str, Any]`

Exports graph for serialization.

**Returns:**
- Serializable dictionary representation

---

### AgentPool

Manages a collection of agents with capabilities and load balancing.

```python
from src.core.agent_pool import AgentPool
```

#### Constructor

```python
AgentPool(agents: Optional[List[Agent]] = None)
```

**Parameters:**
- `agents`: Initial list of agents (optional)

#### Methods

##### `add_agent(agent: Agent) -> None`

Adds an agent to the pool.

##### `get_capable_agents(required_capabilities: np.ndarray) -> List[Agent]`

Returns agents capable of handling given requirements.

##### `get_least_loaded_agent(agents: List[Agent]) -> Agent`

Returns agent with lowest current load.

---

## Coordination Layer

### GNNCoordinator

Central reasoning component using graph neural networks for task assignment.

```python
from src.coordination.gnn_coordinator import GNNCoordinator
```

#### Constructor

```python
GNNCoordinator(
    spectral_norm_bound: float = 0.7,
    temperature: float = 1.0,
    embedding_dim: int = 64,
    gnn_layers: int = 2,
    use_torch: bool = True
)
```

**Parameters:**
- `spectral_norm_bound`: Contractivity bound (must be < 1.0)
- `temperature`: Softmax temperature for assignment randomness
- `embedding_dim`: Node embedding dimension
- `gnn_layers`: Number of GNN layers
- `use_torch`: Whether to use Py