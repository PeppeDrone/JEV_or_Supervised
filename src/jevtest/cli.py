"""Single entry point: stages for one dataset x outcome. GPU stages cache per unit and resume."""
import argparse, sys

import numpy as np

from jevtest.config import GateError
from jevtest.data import load


def check(run):
    counts = np.bincount(run.y, minlength=len(run.classes))
    print(f"{run.dataset}/{run.outcome}: {len(run.y)} labelled patients, {len(run.ds['predictors'])} predictors, "
          f"classes {dict(zip(run.classes, counts.tolist()))}, {len(run.splits)} outer folds -> {run.out}")


def ensemble(run):
    from jevtest.ensembles import run_ensembles

    run_ensembles(run)


def _retired_probe(run):
    print("embed/probe_h retired; skipping", flush=True)


def xgb_holdout(run):
    from jevtest.xgb_optuna import run_holdout

    run_holdout(run)


def xgb_holdout_sensitivity(run):
    from jevtest.xgb_optuna import run_sensitivity

    run_sensitivity(run)


def xgb_holdout_full(run):
    from jevtest.xgb_optuna import run_holdout_full

    run_holdout_full(run)


def xgb_holdout_rescore(run):
    """Reuse the tuned hold-out arm from JEVTEST_HOLDOUT_SOURCE (another study's outputs dir) for this backbone."""
    import os

    from jevtest.xgb_optuna import rescore_holdout

    source = os.environ.get("JEVTEST_HOLDOUT_SOURCE")
    if not source:
        raise GateError("xgb_holdout_rescore needs JEVTEST_HOLDOUT_SOURCE=<outputs dir of the tuned study>")
    rescore_holdout(run, source)


def stages():
    from jevtest import classical, icl, report, zeroshot
    return {
        "check": check,
        "context": icl.plan_context,
        "classical": classical.run_classical,
        "zeroshot": zeroshot.run_zero_shot,
        "embed": _retired_probe,
        "probe_h": _retired_probe,
        "icl": icl.run_icl,
        "ensemble": ensemble,
        "report": report.summarise,
        # Hold-out arms are not part of "all": they run from scripts/queue_xgb_holdout.sh.
        "xgb_holdout": xgb_holdout,
        "xgb_holdout_sensitivity": xgb_holdout_sensitivity,
        "xgb_holdout_full": xgb_holdout_full,
        "xgb_holdout_rescore": xgb_holdout_rescore,
    }


def main(argv=None):
    table = stages()
    parser = argparse.ArgumentParser(prog="jevtest")
    parser.add_argument("command", choices=[*table, "all"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--limit", type=int, help="smoke run on a stratified subset; separate output folder")
    parser.add_argument("--full", action="store_true", help="whole table (patients_full.csv); XGBoost only")
    parser.add_argument("--subset", choices=["all", "holdout"], default="all",
                        help="zeroshot only: 'holdout' scores just the XGB_optuna hold-out patients (large backbones)")
    args = parser.parse_args(argv)
    try:
        run = load(args.root, args.dataset, args.outcome, args.limit, args.full)
        if args.full and args.command not in ("check", "classical", "report"):
            raise GateError("--full is for the XGBoost arm: check, classical, report")
        if args.subset != "all" and args.command != "zeroshot":
            raise GateError("--subset is for the zeroshot stage only")
        standalone = ("xgb_holdout", "xgb_holdout_sensitivity", "xgb_holdout_full", "xgb_holdout_rescore")
        for name in ([n for n in table if n not in standalone] if args.command == "all" else [args.command]):
            if name == "zeroshot" and args.subset == "holdout":
                from jevtest.zeroshot import run_zero_shot
                run_zero_shot(run, subset="holdout")
            else:
                table[name](run)
    except GateError as error:
        print(f"QUALITY GATE: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
