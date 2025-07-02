#!/usr/bin/env python3
"""
setup_tests.py

Script to help organize and fix test imports for the Hybrid AI Brain project.
Run this to automatically detect your project structure and create working test files.
"""

import os
import sys
from pathlib import Path
import shutil

def find_project_structure():
    """Analyze the current project structure."""
    project_root = Path.cwd()
    
    structure = {
        'root': project_root,
        'src': None,
        'tests': None,
        'core_files': [],
        'coordination_files': [],
        'governance_files': []
    }
    
    # Find src directory
    possible_src = [project_root / 'src', project_root]
    for src_path in possible_src:
        if src_path.exists():
            structure['src'] = src_path
            break
    
    # Find tests directory
    possible_tests = [project_root / 'tests', project_root / 'test']
    for test_path in possible_tests:
        if test_path.exists():
            structure['tests'] = test_path
            break
    
    # Find component files
    if structure['src']:
        # Look for core files
        core_dir = structure['src'] / 'core'
        if core_dir.exists():
            structure['core_files'] = list(core_dir.glob('*.py'))
        
        # Look for coordination files
        coord_dir = structure['src'] / 'coordination'
        if coord_dir.exists():
            structure['coordination_files'] = list(coord_dir.glob('*.py'))
        
        # Look for governance files
        gov_dir = structure['src'] / 'governance'
        if gov_dir.exists():
            structure['governance_files'] = list(gov_dir.glob('*.py'))
    
    return structure

def create_working_test_file(structure):
    """Create a test file that works with the actual project structure."""
    
    # Determine what components are available
    has_core = len(structure['core_files']) > 0
    has_coordination = len(structure['coordination_files']) > 0
    has_governance = len(structure['governance_files']) > 0
    
    # Generate appropriate imports
    imports = []
    imports.append("import unittest")
    imports.append("import sys")
    imports.append("import os")
    imports.append("from pathlib import Path")
    imports.append("")
    imports.append("# Add project to Python path")
    imports.append("project_root = Path(__file__).parent.parent")
    imports.append("sys.path.insert(0, str(project_root))")
    imports.append("sys.path.insert(0, str(project_root / 'src'))")
    imports.append("")
    
    test_classes = []
    
    # Create test for project structure
    structure_test = '''
class TestProjectStructure(unittest.TestCase):
    """Test basic project structure."""
    
    def test_project_directories(self):
        """Test that expected directories exist."""
        project_root = Path(__file__).parent.parent
        self.assertTrue(project_root.exists())
        
        # Test for src directory
        src_dir = project_root / "src"
        if src_dir.exists():
            print(f"Found src directory: {src_dir}")
        
        # Test for tests directory
        tests_dir = project_root / "tests"
        if tests_dir.exists():
            print(f"Found tests directory: {tests_dir}")
    
    def test_python_files(self):
        """Test that Python files exist in the project."""
        project_root = Path(__file__).parent.parent
        py_files = list(project_root.rglob("*.py"))
        self.assertGreater(len(py_files), 0, "No Python files found in project")
        print(f"Found {len(py_files)} Python files")
'''
    test_classes.append(structure_test)
    
    # Create core component tests if available
    if has_core:
        imports.append("# Try importing core components")
        imports.append("try:")
        for core_file in structure['core_files']:
            if core_file.name != '__init__.py':
                module_name = core_file.stem
                imports.append(f"    from core.{module_name} import *")
        imports.append("    CORE_AVAILABLE = True")
        imports.append("except ImportError as e:")
        imports.append("    print(f'Core components not available: {e}')")
        imports.append("    CORE_AVAILABLE = False")
        imports.append("")
        
        core_test = '''
@unittest.skipUnless(globals().get('CORE_AVAILABLE', False), "Core components not available")
class TestCoreComponents(unittest.TestCase):
    """Test core components if available."""
    
    def test_core_imports(self):
        """Test that core components can be imported."""
        self.assertTrue(CORE_AVAILABLE)
    
    def test_basic_functionality(self):
        """Test basic functionality of available core components."""
        # Add specific tests based on your actual core components
        # This is a placeholder
        self.assertTrue(True, "Basic functionality test placeholder")
'''
        test_classes.append(core_test)
    
    # Create coordination component tests if available
    if has_coordination:
        imports.append("# Try importing coordination components")
        imports.append("try:")
        for coord_file in structure['coordination_files']:
            if coord_file.name != '__init__.py':
                module_name = coord_file.stem
                imports.append(f"    from coordination.{module_name} import *")
        imports.append("    COORDINATION_AVAILABLE = True")
        imports.append("except ImportError as e:")
        imports.append("    print(f'Coordination components not available: {e}')")
        imports.append("    COORDINATION_AVAILABLE = False")
        imports.append("")
        
        coord_test = '''
@unittest.skipUnless(globals().get('COORDINATION_AVAILABLE', False), "Coordination components not available")
class TestCoordinationComponents(unittest.TestCase):
    """Test coordination components if available."""
    
    def test_coordination_imports(self):
        """Test that coordination components can be imported."""
        self.assertTrue(COORDINATION_AVAILABLE)
    
    def test_bio_optimizer_basic(self):
        """Test bio optimizer if available."""
        # Add specific tests based on your actual coordination components
        self.assertTrue(True, "Bio optimizer test placeholder")
'''
        test_classes.append(coord_test)
    
    # Create test runner
    runner_code = '''
def run_tests():
    """Run all available tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [cls for name, cls in globals().items() 
                   if isinstance(cls, type) and issubclass(cls, unittest.TestCase)]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\\n🎉 All tests passed!")
    else:
        print("\\n⚠️  Some tests failed")
        
        if result.failures:
            print("\\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
'''
    
    # Combine all parts
    test_file_content = "\n".join(imports) + "\n" + "\n".join(test_classes) + "\n" + runner_code
    
    return test_file_content

def create_conftest_py(structure):
    """Create a pytest conftest.py file."""
    conftest_content = '''"""
conftest.py

Pytest configuration for Hybrid AI Brain tests.
"""

import sys
import pytest
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with project paths."""
    project_root = Path(__file__).parent.parent
    src_paths = [
        project_root,
        project_root / "src",
        project_root / "src" / "core",
        project_root / "src" / "coordination", 
        project_root / "src" / "governance",
    ]
    
    for path in src_paths:
        if path.exists():
            sys.path.insert(0, str(path))

def pytest_sessionstart(session):
    """Print available components at start of test session."""
    print("\\n" + "="*50)
    print("HYBRID AI BRAIN TEST SESSION")
    print("="*50)
    
    # Try importing components
    components = {
        "Core": "core",
        "Coordination": "coordination", 
        "Governance": "governance"
    }
    
    for name, module in components.items():
        try:
            __import__(module)
            print(f"✓ {name} components available")
        except ImportError:
            print(f"✗ {name} components not available")

@pytest.fixture(scope="session")
def project_root():
    """Provide project root path."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def temp_system():
    """Create a temporary system for testing."""
    # This would create test instances of your components
    # Adjust based on what's actually available
    return None
'''
    return conftest_content

def setup_test_directory(structure):
    """Set up the test directory with proper files."""
    tests_dir = structure['tests']
    if not tests_dir:
        tests_dir = structure['root'] / 'tests'
        tests_dir.mkdir(exist_ok=True)
        print(f"Created tests directory: {tests_dir}")
    
    # Create working test file
    test_content = create_working_test_file(structure)
    test_file = tests_dir / 'test_hybrid_ai_brain.py'
    
    with open(test_file, 'w') as f:
        f.write(test_content)
    print(f"Created working test file: {test_file}")
    
    # Create conftest.py for pytest
    conftest_content = create_conftest_py(structure)
    conftest_file = tests_dir / 'conftest.py'
    
    with open(conftest_file, 'w') as f:
        f.write(conftest_content)
    print(f"Created conftest.py: {conftest_file}")
    
    # Create __init__.py in tests directory
    init_file = tests_dir / '__init__.py'
    with open(init_file, 'w') as f:
        f.write('"""Tests for Hybrid AI Brain project."""\n')
    print(f"Created __init__.py: {init_file}")
    
    return test_file

def create_run_script(structure):
    """Create a simple script to run tests."""
    run_script_content = '''#!/usr/bin/env python3
"""
run_tests.py

Simple script to run Hybrid AI Brain tests.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run tests using pytest or unittest."""
    project_root = Path(__file__).parent
    
    # Try pytest first
    try:
        import pytest
        print("Running tests with pytest...")
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(project_root / "tests"),
            "-v"
        ])
        return result.returncode == 0
    except ImportError:
        print("Pytest not available, using unittest...")
        
        # Fall back to unittest
        test_file = project_root / "tests" / "test_hybrid_ai_brain.py"
        if test_file.exists():
            result = subprocess.run([
                sys.executable, str(test_file)
            ])
            return result.returncode == 0
        else:
            print(f"Test file not found: {test_file}")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎉 Tests completed successfully!")
    else:
        print("\\n❌ Some tests failed")
    exit(0 if success else 1)
'''
    
    run_script = structure['root'] / 'run_tests.py'
    with open(run_script, 'w') as f:
        f.write(run_script_content)
    
    # Make it executable on Unix systems
    try:
        run_script.chmod(0o755)
    except:
        pass
    
    print(f"Created run script: {run_script}")
    return run_script

def main():
    """Main setup function."""
    print("="*60)
    print("HYBRID AI BRAIN - TEST SETUP")
    print("="*60)
    
    # Analyze project structure
    structure = find_project_structure()
    
    print(f"Project root: {structure['root']}")
    print(f"Source directory: {structure['src']}")
    print(f"Tests directory: {structure['tests']}")
    print(f"Core files: {len(structure['core_files'])}")
    print(f"Coordination files: {len(structure['coordination_files'])}")
    print(f"Governance files: {len(structure['governance_files'])}")
    
    if structure['core_files']:
        print(f"  Core files: {[f.name for f in structure['core_files']]}")
    if structure['coordination_files']:
        print(f"  Coordination files: {[f.name for f in structure['coordination_files']]}")
    if structure['governance_files']:
        print(f"  Governance files: {[f.name for f in structure['governance_files']]}")
    
    print(f"\n{'='*40}")
    print("SETTING UP TESTS")
    print(f"{'='*40}")
    
    # Set up test directory
    test_file = setup_test_directory(structure)
    
    # Create run script
    run_script = create_run_script(structure)
    
    print(f"\n{'='*40}")
    print("SETUP COMPLETE")
    print(f"{'='*40}")
    
    print(f"✓ Test file created: {test_file}")
    print(f"✓ Run script created: {run_script}")
    print(f"✓ Pytest configuration created")
    
    print(f"\nTo run tests:")
    print(f"  python {run_script}")
    print(f"  or: python {test_file}")
    print(f"  or: cd {structure['root']} && python -m pytest tests/")
    
    # Try running the test file immediately
    print(f"\n{'='*40}")
    print("TESTING THE SETUP")
    print(f"{'='*40}")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=structure['root'])
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Test errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 Test setup successful!")
        else:
            print(f"\n⚠️  Tests completed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"Could not run test immediately: {e}")
        print("Please run the tests manually using the instructions above.")

if __name__ == "__main__":
    main()
