#!/usr/bin/env python3
"""Rebuild per-outcome figures and combined overview after an outcome updates.

Usage:
  refresh_outcome_figures.py --dataset uci_ckd --outcome ckd_status
  refresh_outcome_figures.py --combined-only
  refresh_outcome_figures.py --watch   # poll predictions; refresh on change
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VENV_PY = ROOT / ".venv" / "bin" / "python"

from jevtest.benchmark import ALL_PAIRS as PAIRS
from jevtest.config import outputs_root

LOG = outputs_root(ROOT) / "logs" / "figure_refresh.log"
STATE = outputs_root(ROOT) / "logs" / "figure_refresh_state.json"


def log(msg: str) -> None:
    # Callers typically redirect stdout to LOG; avoid double-writing.
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


def pred_fingerprint(dataset: str, outcome: str) -> str:
    pred = outputs_root(ROOT) / dataset / outcome / "predictions"
    if not pred.is_dir():
        return ""
    parts = []
    for path in sorted(pred.glob("*.csv")):
        st = path.stat()
        parts.append(f"{path.name}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest() if parts else ""


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=ROOT)


def refresh(dataset: str | None, outcome: str | None, combined: bool, complexity: bool) -> None:
    py = str(VENV_PY if VENV_PY.exists() else sys.executable)
    env_prefix = [py]
    if dataset and outcome:
        log(f"REFRESH {dataset}/{outcome}")
        code = run([*env_prefix, "scripts/plot_predictions.py",
                    "--dataset", dataset, "--outcome", outcome, "--no-combined"])
        if code != 0:
            log(f"FAIL outcome plot {dataset}/{outcome} exit={code}")
    if combined:
        log("REFRESH combined")
        code = run([*env_prefix, "scripts/plot_predictions.py", "--combined-only"])
        if code != 0:
            log(f"FAIL combined plot exit={code}")
        else:
            log("OK combined")
    if complexity:
        log("REFRESH complexity")
        code = run([*env_prefix, "scripts/analyse_uci_complexity_gaps.py"])
        if code != 0:
            log(f"FAIL complexity exit={code}")
        else:
            log("OK complexity")
        code = run([*env_prefix, "scripts/summarise_uci_core.py"])
        if code != 0:
            log(f"FAIL summarise exit={code}")


def watch(poll: float, settle: float, with_complexity: bool = True) -> None:
    import json

    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text()).get("fps", {})
        except json.JSONDecodeError:
            state = {}

    # Seed without rebuilding everything on first start
    for ds, oc in PAIRS:
        key = f"{ds}/{oc}"
        state.setdefault(key, pred_fingerprint(ds, oc))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"fps": state}, indent=2) + "\n")
    log(f"WATCH start poll={poll}s settle={settle}s pairs={len(PAIRS)} "
        f"complexity={'on' if with_complexity else 'off'}")

    pending: dict[str, float] = {}
    while True:
        time.sleep(poll)
        now = time.time()
        for ds, oc in PAIRS:
            key = f"{ds}/{oc}"
            fp = pred_fingerprint(ds, oc)
            if not fp:
                continue
            if fp != state.get(key):
                pending[key] = now
                state[key] = fp
                log(f"CHANGE detected {key}")

        ready = [k for k, t0 in pending.items() if now - t0 >= settle]
        for key in ready:
            ds, oc = key.split("/", 1)
            # Prefer refreshing once Probe (or at least M2) exists
            pred = outputs_root(ROOT) / ds / oc / "predictions"
            if not (pred / "M2.csv").exists():
                continue
            del pending[key]
            refresh(ds, oc, combined=True, complexity=with_complexity)
            STATE.write_text(json.dumps({"fps": state}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--outcome")
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--with-complexity", action="store_true", default=None,
                        help="Rebuild complexity logistic figures (default: on for --watch)")
    parser.add_argument("--no-complexity", action="store_true",
                        help="Skip complexity rebuild even in --watch mode")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=float, default=45.0)
    parser.add_argument("--settle", type=float, default=20.0,
                        help="Seconds to wait after last prediction change before plotting")
    args = parser.parse_args()

    if args.no_complexity:
        with_complexity = False
    elif args.with_complexity:
        with_complexity = True
    else:
        # Watch defaults to refreshing complexity; one-shot calls opt in explicitly.
        with_complexity = bool(args.watch)

    if args.watch:
        watch(args.poll, args.settle, with_complexity=with_complexity)
        return

    if args.combined_only and not (args.dataset and args.outcome):
        refresh(None, None, combined=True, complexity=with_complexity)
        return

    if not args.dataset or not args.outcome:
        parser.error("Need --dataset/--outcome, --combined-only, or --watch")
    refresh(args.dataset, args.outcome, combined=True, complexity=with_complexity)


if __name__ == "__main__":
    main()
