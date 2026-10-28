#!/usr/bin/env python3
"""Print benchmark completion snapshot for periodic status updates."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS as PAIRS
from jevtest.config import outputs_root

STATE = outputs_root(ROOT) / "logs" / "benchmark_progress_state.json"
MODELS = ["M2", "ZS_raw"]
ESSENTIAL = {"M2", "ZS_raw"}


def snapshot():
    rows = {}
    for ds, oc in PAIRS:
        pred = outputs_root(ROOT) / ds / oc / "predictions"
        have = [m for m in MODELS if (pred / f"{m}.csv").exists()]
        key = f"{ds}/{oc}"
        rows[key] = {
            "have": have,
            "classical": {"M2"} <= set(have),
            "full": ESSENTIAL <= set(have),
        }
    return rows


def main() -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    cur = snapshot()
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text()).get("rows", {})
        except json.JSONDecodeError:
            prev = {}

    newly_classical = []
    newly_full = []
    for key, info in cur.items():
        old = prev.get(key, {})
        if info["classical"] and not old.get("classical"):
            newly_classical.append(key)
        if info["full"] and not old.get("full"):
            newly_full.append(key)

    n = len(PAIRS)
    n_classical = sum(1 for v in cur.values() if v["classical"])
    n_full = sum(1 for v in cur.values() if v["full"])
    print(f"[{now}] Benchmark progress: classical {n_classical}/{n}, full(M2+ZS_raw) {n_full}/{n}")
    if newly_full:
        print("Newly finished (full):")
        for k in newly_full:
            print(f"  + {k} [{','.join(cur[k]['have'])}]")
    if newly_classical:
        print("Newly finished classical (M2):")
        for k in newly_classical:
            if k not in newly_full:
                print(f"  + {k}")
    if not newly_full and not newly_classical:
        print("No new completions since last check.")

    still = [k for k, v in cur.items() if not v["full"]]
    print(f"Still incomplete ({len(still)}):")
    for k in still:
        have = cur[k]["have"]
        print(f"  - {k} [{','.join(have) if have else 'none'}]")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"updated": now, "rows": cur}, indent=2) + "\n")


if __name__ == "__main__":
    main()
