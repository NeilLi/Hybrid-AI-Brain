#!/usr/bin/env python3
"""
src/integrations/langgraph_adapter.py

Adapter to translate Hybrid AI Brain's task graph into a LangGraph StateGraph.
"""

from typing import Dict, Any, Optional, List, Callable

class BrainTaskGraph:
    """Placeholder for the TaskGraph produced by Hybrid AI Brain."""
    def __init__(self):
        self.nodes = {
            "fetch_data": {"type": "tool_call", "tool": "data_fetcher"},
            "analyze_sentiment": {"type": "agent_task", "agent": "nlp_agent"},
            "assess_risk": {"type": "agent_task", "agent": "risk_agent"},
            "final_summary": {"type": "llm_call"},
        }
        self.dependencies = [
            ("fetch_data", "analyze_sentiment"),
            ("fetch_data", "assess_risk"),
            ("analyze_sentiment", "final_summary"),
            ("assess_risk", "final_summary"),
        ]
        print("BrainTaskGraph (Placeholder): Initialized with a sample task structure.")

class LangGraphAdapter:
    """
    Adapts a Hybrid AI Brain TaskGraph to a LangGraph StateGraph.
    """
    def __init__(self, brain_task_graph: BrainTaskGraph):
        self.brain_task_graph = brain_task_graph
        print("LangGraphAdapter initialized.")

    def to_langgraph_state_machine(self) -> Optional[Any]:
        """Compiles the brain's graph into a LangGraph StateGraph."""
        try:
            from langgraph.graph import StateGraph, END
            from typing import TypedDict, List
        except ImportError:
            print("Warning: langgraph library not found. Please 'pip install langgraph'.")
            return None

        print("\nLangGraphAdapter: Compiling brain's task graph into a StateGraph...")

        class GraphState(TypedDict):
            steps: List[str]
            results: Dict[str, Any]

        workflow = StateGraph(GraphState)

        # --- Closure-safe node function factory ---
        def make_node_function(task_id: str) -> Callable[[GraphState], GraphState]:
            def node_function(state: GraphState) -> GraphState:
                print(f"  - LangGraph Node Executing: '{task_id}'")
                current_steps = state.get("steps", [])
                current_results = state.get("results", {})
                current_steps.append(task_id)
                current_results[task_id] = f"Result from {task_id}"
                return {"steps": current_steps, "results": current_results}
            return node_function

        # Add nodes (each with a unique function)
        for node_id, node_info in self.brain_task_graph.nodes.items():
            workflow.add_node(node_id, make_node_function(node_id))
            print(f"  - Added node: '{node_id}'")

        # Add edges for all dependencies
        for start_node, end_node in self.brain_task_graph.dependencies:
            if start_node not in self.brain_task_graph.nodes or end_node not in self.brain_task_graph.nodes:
                raise ValueError(f"Dependency references unknown node: {start_node}->{end_node}")
            workflow.add_edge(start_node, end_node)
            print(f"  - Added edge: '{start_node}' -> '{end_node}'")

        # Compute entry points (nodes with no incoming edges)
        all_nodes = set(self.brain_task_graph.nodes)
        dest_nodes = {dst for _, dst in self.brain_task_graph.dependencies}
        entry_nodes = list(all_nodes - dest_nodes)
        if not entry_nodes:
            raise ValueError("No entry point found in the task graph.")
        workflow.set_entry_point(entry_nodes[0])
        print(f"  - Set entry point: '{entry_nodes[0]}'")

        # Add edges to END for all terminal nodes (no outgoing edges)
        source_nodes = {src for src, _ in self.brain_task_graph.dependencies}
        for node_id in all_nodes:
            if node_id not in source_nodes:
                workflow.add_edge(node_id, END)
                print(f"  - Added edge to END from terminal node: '{node_id}'")

        app = workflow.compile()
        print("\nLangGraphAdapter: Compilation successful.")
        return app

def main():
    print("====== Integrations Layer: LangGraphAdapter Demo ======")
    task_graph = BrainTaskGraph()
    adapter = LangGraphAdapter(task_graph)
    langgraph_app = adapter.to_langgraph_state_machine()
    if langgraph_app:
        print("\n--- Executing the compiled LangGraph App ---")
        initial_state = {"steps": [], "results": {}}
        final_state = langgraph_app.invoke(initial_state)
        print("\n--- LangGraph Execution Complete ---")
        import json
        print(json.dumps(final_state, indent=2))
    print("\n====================================================")
    print("✅ langgraph_adapter.py executed successfully!")

if __name__ == "__main__":
    main()
