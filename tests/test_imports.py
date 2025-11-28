
import sys
import os
import importlib
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


MODULES = [
    "src.training.train_model",
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
