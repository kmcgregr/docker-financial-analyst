"""
Import tests — verify every module loads without error.

NOTE: These tests require all project dependencies installed
(langchain, chromadb, PyMuPDF, etc.). Run:
  pip install -r requirements.txt
"""

import importlib
import pytest

MODULES = [
    "utils",
    "tasks",
    "vision_extractor",
    "valuation_rag",
    "orchestrator",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Each project module imports without raising ImportError."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.skip("Install deps: pip install -r requirements.txt")
