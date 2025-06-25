#!/usr/bin/env python3
"""
benchmarks/__init__.py

Makes the benchmarks directory a Python package.
This allows for easier importing of benchmark setup functions
(like setup_fifa_scenario) into other test or benchmark scripts.
"""

from .fifa_scenario import setup_fifa_scenario
from .synthetic_tasks import generate_synthetic_task_graph

__all__ = [
    "setup_fifa_scenario",
    "generate_synthetic_task_graph",
]
