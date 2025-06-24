#!/usr/bin/env python3
"""
setup.py

A setuptools-based setup script for the Hybrid AI Brain project.
"""

from setuptools import setup, find_packages
from pathlib import Path

PACKAGE_NAME = "hybrid_ai_brain"
VERSION = "1.0.0"
AUTHOR = "Ning Li"
AUTHOR_EMAIL = "ning.li@example.com"
DESCRIPTION = "Hybrid AI Brain: Multi-agent micro-cell AI with GNN coordination and provable safety, convergence, and memory guarantees."
URL = "https://github.com/NeilLi/Hybrid-AI-Brain"  # Update as needed


def get_long_description() -> str:
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return DESCRIPTION


def get_requirements() -> list:
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        print("Warning: requirements.txt not found. Installing without dependencies.")
        return []


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Paper": "https://arxiv.org/abs/your-paper-id",  # Optional: update with your actual paper link
        "Documentation": "https://github.com/NeilLi/Hybrid-AI-Brain/tree/main/docs",
    },
    keywords=[
        "multi-agent",
        "GNN",
        "swarm intelligence",
        "AI",
        "memory",
        "convergence",
        "safety",
        "python",
        "autogen",
        "langgraph",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    extras_require={
        "chromadb": ["chromadb>=0.4.0", "sentence-transformers>=2.2.0"],
        "retrieval": ["chromadb>=0.4.0", "sentence-transformers>=2.2.0"],
        "gpu": ["torch>=2.0.0"],  # for CUDA/GPU users
        "dev": ["pytest", "black", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "hybrid-brain=hybrid_ai_brain.cli:main",  # This assumes you’ll implement cli.py
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
