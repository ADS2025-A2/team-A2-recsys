
import sys
import os
import importlib
import pytest
import sys
from pathlib import Path
    
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "training"))
print("PROJECT_ROOT:", PROJECT_ROOT)
#print("DIRS in project root: ", list(PROJECT_ROOT.parent.iterdir()), file=sys.stderr)
sys.path.insert(0, PROJECT_ROOT)


MODULES = [
    "train_model"
]

@pytest.mark.parametrize("module", MODULES)
def test_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        pytest.fail(f"Module not found: {module} ({e})")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {module}: {e}")

