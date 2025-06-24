#!/usr/bin/env python3
"""
src/integrations/autogen_adapter.py

Adapter for integrating Hybrid AI Brain as an AutoGen ConversableAgent.
"""

from typing import Dict, Any, List, Optional, Callable

class BrainOrchestrator:
    """A placeholder for the Hybrid AI Brain orchestrator."""
    def process_request(self, request: str) -> Dict[str, str]:
        print(f"\n--- Hybrid AI Brain: Processing request: '{request}' ---")
        print("--- Coordination complete (simulated). ---")
        return {"task_1": "agent_A", "task_2": "agent_B"}

class AutoGenAdapter:
    """
    Wraps Hybrid AI Brain as a ConversableAgent in AutoGen, allowing agentic interaction.
    """
    def __init__(
        self,
        brain_orchestrator: BrainOrchestrator,
        name: str = "Hybrid_AI_Brain",
        system_message: Optional[str] = None,
        llm_config: bool = False,
    ):
        self.brain_orchestrator = brain_orchestrator
        self.agent = self._create_agent(name, system_message, llm_config)
        print(f"AutoGenAdapter initialized (agent name: {name})")

    def _create_agent(
        self,
        name: str,
        system_message: Optional[str],
        llm_config: bool,
    ) -> Optional[Any]:
        try:
            from autogen import ConversableAgent
        except ImportError:
            print("Warning: autogen library not found. Please 'pip install autogen-agentchat'.")
            return None

        def generate_brain_reply(
            messages: List[Dict[str, Any]],
            sender: Any,
            **kwargs
        ) -> Dict[str, Any]:
            # Expect the user’s request as the last message content.
            request = messages[-1]["content"]
            assignments = self.brain_orchestrator.process_request(request)
            response_str = (
                "Hybrid AI Brain has processed the request. "
                f"Optimal assignment plan: {assignments}"
            )
            return {"content": response_str, "role": "assistant"}

        agent = ConversableAgent(
            name=name,
            system_message=system_message or (
                "I am a specialized multi-agent coordination system. "
                "I receive complex tasks and return an optimal, safe "
                "execution plan for my internal agent swarm."
            ),
            llm_config=llm_config,
        )
        agent.register_reply(
            trigger=ConversableAgent,
            reply_func=generate_brain_reply,
            position=1,
        )
        print("  - AutoGen ConversableAgent for Hybrid AI Brain created.")
        return agent

def main():
    print("====== Integrations Layer: AutoGenAdapter Demo ======")
    brain = BrainOrchestrator()
    adapter = AutoGenAdapter(brain)

    if adapter.agent:
        try:
            from autogen import UserProxyAgent
            user_proxy = UserProxyAgent(
                name="User",
                code_execution_config=False,
                human_input_mode="NEVER",
                default_auto_reply="Thank you, this is complete."
            )
            print("\n--- Simulating a User Task via AutoGen ---")
            user_proxy.initiate_chat(
                recipient=adapter.agent,
                message="Decompose the query 'What is the GDP per capita of the "
                        "country that won the most recent FIFA World Cup?' "
                        "and assign the subtasks.",
                max_turns=2,
            )
        except ImportError:
            print("\n[Warning] Demo chat skipped: autogen package is not installed.")
    print("\n====================================================")
    print("✅ autogen_adapter.py executed successfully!")

if __name__ == "__main__":
    main()
