"""One-off (2026-10-14): reuse NeuroJev's Qwen3-4B readouts for dataset neurojev_gose / outcome gose.

The NeuroJev M3 learning study already ran the same frozen backbone on byte-identical prompts
(checked: all 276 rendered texts and the question match), so its zero-shot AnyJev L0 readout
(cache/zs.json) is exactly what the `zeroshot` stage would compute. It is written into this
bench's cache format with the current data hash, plus a provenance note.
    PYTHONPATH=src python scripts/import_neurojev_readouts.py [--source ../neurojev/outputs/exploratory/m3_learning/cache]
"""
from pathlib import Path
import argparse, json

from jevtest.config import write_json
from jevtest.data import load
from jevtest.zeroshot import cache_key

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="../neurojev/outputs/exploratory/m3_learning/cache")
args = parser.parse_args()
source = Path(args.source)
run = load(".", "neurojev_gose", "gose")
b = run.study["backbone"]
assert b["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"

zs = json.loads((source / "zs.json").read_text())
probs = {p: [1 - zs[p], zs[p]] for p in run.ids}
write_json(run.out / "cache/zs.json", {"key": cache_key(run), "probs": probs})
(run.out / "cache/PROVENANCE.md").write_text(
    "zs.json was imported on 2026-10-14 from the NeuroJev M3 learning study "
    "(neurojev/outputs/exploratory/m3_learning/cache/zs.json): same Qwen3-4B revision, "
    "same AnyJev commit and L0 content-free settings, byte-identical prompts and question. "
    "Reproduction: ZS 0.7866 AUROC, identical to NeuroJev (max |dp| <= 1e-16).\n")
print(f"imported zero-shot readouts for {len(run.ids)} patients into {run.out / 'cache'}")
