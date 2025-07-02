"""
tests/conftest.py
Pytest configuration to fix import paths for both local development and CI/CD
"""
import sys
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent
paths_to_add = [
    project_root,
    project_root / 'src',
    project_root / 'src' / 'core',
    project_root / 'src' / 'coordination', 
    project_root / 'src' / 'governance'
]

for path in paths_to_add:
    if path.exists():
        sys.path.insert(0, str(path))

def pytest_configure(config):
    """Configure pytest with debug info."""
    print(f"\nProject root: {project_root}")
    print(f"Added paths: {[str(p) for p in paths_to_add if p.exists()]}")