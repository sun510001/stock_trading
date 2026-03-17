from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / ".private_data"
    / "tools"
    / "processing_tools"
    / "regime_dataset_builder.py"
)


def _load_builder_class() -> type:
    """Load the private regime dataset builder implementation."""
    spec = importlib.util.spec_from_file_location(
        "trendalloc_private_regime_dataset_builder",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load regime dataset builder from {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    builder_class = getattr(module, "RegimeDatasetBuilder", None)
    if builder_class is None:
        raise ImportError(
            f"RegimeDatasetBuilder is not defined in {MODULE_PATH}"
        )
    return builder_class


RegimeDatasetBuilder = _load_builder_class()
