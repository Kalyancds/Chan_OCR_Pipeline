"""Ensure the project root is importable so ``import ocr_qa`` works from a
plain ``pytest`` invocation (no PYTHONPATH needed)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
