#!/usr/bin/env python3
"""
src/integrations/external_agents.py

Manager for registering and interacting with external tools, APIs, or agentic systems.
"""

from typing import Dict, Any, Callable, Optional, List
import logging
import threading

logger = logging.getLogger("hybrid_ai_brain.external_agents")
logging.basicConfig(level=logging.INFO)

class ExternalAgentManager:
    """
    Manages a registry of external tools and agents.
    Allows the Hybrid AI Brain to incorporate capabilities beyond its core swarm.
    """
    def __init__(self):
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()
        logger.info("ExternalAgentManager initialized.")

    def register_tool(self, tool_name: str, tool_function: Callable[..., Any]) -> None:
        """Registers a new external tool or agent."""
        with self._lock:
            if tool_name in self._registry:
                logger.warning(f"Overwriting existing tool in registry: '{tool_name}'")
            self._registry[tool_name] = tool_function
            logger.info(f"Registered external tool: '{tool_name}'")

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Calls a registered external tool with the given arguments.
        Returns the result or an error dictionary.
        """
        logger.info(f"Calling tool '{tool_name}' with args: {kwargs}")
        with self._lock:
            tool_function = self._registry.get(tool_name)
        if not tool_function:
            error_message = f"Tool '{tool_name}' not found in the registry."
            logger.error(error_message)
            return {"error": error_message}
        try:
            result = tool_function(**kwargs)
            logger.info(f"Tool '{tool_name}' executed successfully.")
            return result
        except Exception as e:
            error_message = f"Error executing tool '{tool_name}': {e}"
            logger.exception(error_message)
            return {"error": error_message}

    def list_tools(self) -> List[str]:
        """Returns a list of all registered external tool names."""
        with self._lock:
            return list(self._registry.keys())

def main():
    """Demonstrates how to register and use external agents/tools."""
    print("====== Integrations Layer: ExternalAgentManager Demo ======")

    # 1. Define some example external tool functions
    def web_search(query: str) -> Dict[str, Any]:
        """A mock function for a web search tool."""
        return {"query": query, "results": [{"title": "Python Website", "url": "python.org"}]}

    def weather_api(city: str) -> Dict[str, Any]:
        """A mock function for a weather API."""
        return {"city": city, "temperature": "25°C", "condition": "Sunny"}

    # 2. Initialize the manager
    external_manager = ExternalAgentManager()

    # 3. Register the tools
    print("\n--- Registering External Tools ---")
    external_manager.register_tool("web_search", web_search)
    external_manager.register_tool("weather_api", weather_api)

    print("Available tools:", external_manager.list_tools())

    # 4. The Hybrid AI Brain's coordinator would now decide to call a tool
    print("\n--- Simulating Brain's decision to call a tool ---")
    
    # Scenario A: The brain needs to search the web
    search_result = external_manager.call_tool("web_search", query="what is python")
    print("Result from web_search:", search_result)

    # Scenario B: The brain needs to get the weather
    weather_result = external_manager.call_tool("weather_api", city="San Francisco")
    print("Result from weather_api:", weather_result)

    # Scenario C: The brain tries to call a non-existent tool
    error_result = external_manager.call_tool("code_interpreter", code="print('hello')")
    print("Result from non-existent tool:", error_result)
    
    print("\n====================================================")
    print("✅ external_agents.py executed successfully!")

if __name__ == "__main__":
    main()
