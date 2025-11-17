import importlib
import pytest


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
