#!/usr/bin/env python3
"""Exit 0 only when every planned 27B v2 inference file is complete; prints per-stage coverage.

Completeness is read from saved outputs (identities in manifest.json versus scored keys), never from exit codes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from jevtest.benchmark import ALL_PAIRS
from jevtest.revision_paths import CURRENT
from jevtest.revision_rendering import CONDITIONS


def main(backbone="27b"):
    base = CURRENT / "inference" / backbone
    missing = {}
    for dataset, outcome in ALL_PAIRS:
        dest = base / dataset / outcome
        if not (dest / "manifest.json").exists():
            for stage in ("manifest", "development", *CONDITIONS, "direction"):
                missing.setdefault(stage, []).append(outcome)
            continue
        meta = json.loads((dest / "manifest.json").read_text())
        planned = {"development": set(meta["development_ids"])} | {c: set(meta["holdout_ids"]) for c in CONDITIONS}
        for stage, ids in planned.items():
            path = dest / f"{stage}.json"
            if not path.exists() or set(json.loads(path.read_text())) != ids:
                missing.setdefault(stage, []).append(outcome)
        plan = dest / "direction_plan.json"
        if not plan.exists():
            missing.setdefault("direction", []).append(outcome)
        else:
            features = json.loads(plan.read_text())
            if features and not (dest / "direction.json").exists():
                missing.setdefault("direction", []).append(outcome)
            if len(features) == 2 and not (dest / "interaction.json").exists():
                missing.setdefault("interaction", []).append(outcome)
    for stage in ("manifest", "development", *CONDITIONS, "direction", "interaction"):
        print(f"{stage}: {len(ALL_PAIRS) - len(missing.get(stage, [])) if stage != 'interaction' else 'n/a'} complete"
              + (f"; incomplete {len(missing[stage])}" if stage in missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
