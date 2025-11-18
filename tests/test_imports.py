
import sys
import os
import importlib
import pytest

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


MODULES = [
    "src.train",
    "src.utils",
]


@pytest.mark.parametrize("module", MODULES)
def test_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        pytest.fail(f"Module not found: {module} ({e})")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {module}: {e}")
