#!/usr/bin/env python3
"""Additional pinned AnyJev inference, separate from all original artifacts.

Example (small cached model; fixed task subset, not selected on test performance):
  .venv/bin/python scripts/run_revision_inference.py --backbone 0.8b \
      --datasets uci_breast_coimbra,uci_hepatitis,uci_dermatology --development --semantic --direction
Full 27B requested run:
  .venv/bin/python scripts/run_revision_inference.py --backbone 27b --development --semantic --direction
Capped 27B design actually executed (prespecified, label-blind; see scripts/queue_revision_27b.sh):
  --dev-pool 200 --holdout-cap 100 run stage by stage (--stages development / original / <condition> / direction).
Use --resume only for the same configuration. Never touches pod run folders.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from jevtest.benchmark import ALL_PAIRS
from jevtest.config import write_json, digest
from jevtest.data import load, predictor_matrix
from jevtest.llm import anyjev_runtime
from jevtest.revision_rendering import CONDITIONS, render_condition, direction_plan
from jevtest.revision_paths import CURRENT

DEFAULT_OUT = CURRENT / "inference"
STAGES = ("development", *CONDITIONS, "direction")


def label_blind_subset(ids, cap, seed):
    """Uniform sample without replacement from a split list, using identities only (never outcomes)."""
    ids = list(ids)
    if not cap or cap >= len(ids):
        return ids
    pick = np.sort(np.random.default_rng(seed).choice(len(ids), cap, replace=False))
    return [ids[i] for i in pick]


def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {a: plain(b) for a, b in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, choices=["0.8b", "2b", "4b", "9b", "27b", "397b_a17b_fp8"])
    ap.add_argument("--datasets", default="")
    ap.add_argument("--development", action="store_true")
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--direction", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dev-pool", type=int, default=0, help="label-blind development calibration pool size (0 = all)")
    ap.add_argument("--holdout-cap", type=int, default=0, help="label-blind hold-out subset for rendering/protocol reads (0 = all)")
    ap.add_argument("--stages", default="", help=f"comma list from {STAGES}; default follows the legacy flags")
    ap.add_argument("--sort-length", action="store_true", help="batch prompts in rendered-length order (padding efficiency)")
    ap.add_argument("--single-device", action="store_true", help="place the whole backbone on cuda:0 (no CPU offload)")
    args = ap.parse_args()
    stages = [a for a in args.stages.split(",") if a]
    if any(a not in STAGES for a in stages):
        raise ValueError(f"unknown stage in {stages}")
    if not stages:
        stages = (["development"] if args.development else []) + ["original"] + \
                 (list(CONDITIONS[1:]) if args.semantic else []) + (["direction"] if args.direction else [])
    study = yaml.safe_load((ROOT / f"configs/study_qwen3.5_{args.backbone}.yaml").read_text())
    # For the FP8 configuration its filename includes the FP8 suffix, as above.
    study["backbone"]["anyjev_batch"] = args.batch
    if args.single_device:
        # Pod configs use device_map=auto for multi-GPU sharding; on unified-memory GB10 the auto map offloads
        # layers to CPU because page cache reduces reported free memory. Pin every module to GPU 0 instead.
        study["backbone"]["device_map"] = {"": 0}
    backend, decider = anyjev_runtime(study)
    from anyjev import Question
    selected = set(args.datasets.split(",")) - {""}
    for dataset, outcome in ALL_PAIRS:
        if selected and dataset not in selected:
            continue
        run = load(ROOT, dataset, outcome)
        split_path = ROOT / "outputs" / dataset / outcome / "holdout/split.json"
        split = json.loads(split_path.read_text())
        index = {pid: i for i, pid in enumerate(run.ids)}
        dev_ids = label_blind_subset(split["development"], args.dev_pool, study["seed"])
        hold_ids = label_blind_subset(split["holdout"], args.holdout_cap, study["seed"] + 1)
        dev, hold = [index[p] for p in dev_ids], [index[p] for p in hold_ids]
        all_dev = [index[p] for p in split["development"]]
        question = Question.choice(run.spec["question"], run.classes, name=outcome)
        dest = args.out / args.backbone / dataset / outcome
        dest.mkdir(parents=True, exist_ok=True)
        key = dict(backbone=study["backbone"], seed=study["seed"], split_sha256=digest(split_path),
                   data_hash=run.data_hash, classes=run.classes,
                   renderer_sha256=digest(ROOT / "src/jevtest/revision_rendering.py"),
                   inference_sha256=digest(Path(__file__)), command=sys.argv)
        if args.dev_pool or args.holdout_cap:
            # Capped design identity: pool sizes and the exact sampled identities.
            key.update(dev_pool=args.dev_pool, holdout_cap=args.holdout_cap, sort_length=args.sort_length,
                       development_ids=[str(p) for p in dev_ids], holdout_ids=[str(p) for p in hold_ids],
                       sampling="numpy default_rng(seed) / (seed+1) uniform without replacement over split order; outcome-blind")
        # Resume is operational, not an experiment identity.
        key.pop("command")
        manifest = dest / "manifest.json"
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text()) != key:
                raise RuntimeError(f"existing inference requires matching --resume: {dest}")
        else:
            write_json(manifest, key)

        def score_group(name, items):
            path = dest / f"{name}.json"
            saved = json.loads(path.read_text()) if path.exists() and args.resume else {}
            pending = [item for item in items if item[0] not in saved]
            if args.sort_length:
                pending.sort(key=lambda item: len(item[1]))
            for offset in range(0, len(pending), args.batch):
                chunk = pending[offset:offset + args.batch]
                decisions = decider.decide_batch([item[1] for item in chunk], question, level="L0", require="L0")
                for item, result in zip(chunk, decisions):
                    saved[item[0]] = dict(probs=plain(result.probs), diagnostics=plain(result.diagnostics), **item[2])
                write_json(path, saved)
                print(f"{args.backbone} {dataset}/{outcome} {name} {len(saved)}/{len(items)}", flush=True)
            return saved

        if "development" in stages:
            score_group("development", [(str(run.ids[i]), render_condition(run.frame.iloc[i], run.ds, "original"), {}) for i in dev])
        # Original hold-out diagnostics are also regenerated; all stage ablations reuse these reads.
        for condition in CONDITIONS:
            if condition in stages:
                score_group(condition, [(str(run.ids[i]), render_condition(run.frame.iloc[i], run.ds, condition), {}) for i in hold])
        if "direction" in stages:
            # Feature choice uses the complete development partition (labels allowed there), never hold-out labels.
            X_dev = predictor_matrix(run).iloc[all_dev]
            plan = direction_plan(X_dev, run.y[all_dev], run.ds, len(run.classes))
            write_json(dest / "direction_plan.json", plan)
            rng = np.random.default_rng(study["seed"])
            patients = rng.choice(all_dev, min(10, len(all_dev)), replace=False)
            items = []
            for feature in plan:
                for i in patients:
                    for level in ("low", "high"):
                        row = run.frame.iloc[i].copy()
                        row[feature["column"]] = feature[level]
                        name = f"{run.ids[i]}::{feature['predictor']}::{level}"
                        items.append((name, render_condition(row, run.ds, "original"), dict(patient_id=str(run.ids[i]), level=level, **feature)))
            score_group("direction", items)
            # Controlled two-feature contrast; interpretation remains limited by unverified joint plausibility.
            if len(plan) == 2:
                interaction = []
                for i in patients:
                    for a in ("low", "high"):
                        for b in ("low", "high"):
                            row = run.frame.iloc[i].copy()
                            row[plan[0]["column"]], row[plan[1]["column"]] = plan[0][a], plan[1][b]
                            interaction.append((f"{run.ids[i]}::{a}::{b}", render_condition(row, run.ds, "original"),
                                                dict(patient_id=str(run.ids[i]), first=a, second=b, contrast=len(run.classes)-1,
                                                     joint_plausibility="unverified; within marginal ranges only")))
                score_group("interaction", interaction)


if __name__ == "__main__":
    main()
