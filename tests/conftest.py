"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Make `src` importable from tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
