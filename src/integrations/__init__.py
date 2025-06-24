#!/usr/bin/env python3
"""
src/integrations/__init__.py

Exposes the primary adapter and manager classes of the integrations layer,
making it a cohesive and importable package.
"""

from .autogen_adapter import AutoGenAdapter
from .langgraph_adapter import LangGraphAdapter
from .external_agents import ExternalAgentManager

__all__ = [
    "AutoGenAdapter",
    "LangGraphAdapter",
    "ExternalAgentManager",
]
