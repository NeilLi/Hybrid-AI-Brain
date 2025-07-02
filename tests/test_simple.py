# tests/test_simple.py
import unittest
import sys
from pathlib import Path

# Add your project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

class TestBasic(unittest.TestCase):
    def test_project_structure(self):
        """Test basic project structure."""
        self.assertTrue(project_root.exists())
        print(f"Project root: {project_root}")
        
        # List what's actually in your src directory
        src_dir = project_root / 'src'
        if src_dir.exists():
            py_files = list(src_dir.rglob('*.py'))
            print(f"Found {len(py_files)} Python files")
            for f in py_files[:10]:  # Show first 10
                print(f"  {f}")

if __name__ == "__main__":
    unittest.main()
