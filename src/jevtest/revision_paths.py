"""Versioned revision result trees. v1 is frozen provenance; new derived results go to the current tree."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "reports/scientific_revision_2026-10-25_v1"
CURRENT = ROOT / "reports/scientific_revision_2026-10-26_v2"
# Additional inference roots, read in this order; a task/backbone may appear in only one of them.
INFERENCE_ROOTS = (PREVIOUS / "inference", CURRENT / "inference")
