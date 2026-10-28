"""Write ``patients_full.csv`` (no row cap) for the ten tables whose ``patients.csv`` is a subsample.

``patients.csv`` is untouched: it keeps feeding the zero-shot arm and the complexity and codebook
analyses. ``patients_full.csv`` is read only by ``jevtest.cli ... --full`` for the XGBoost arm.
Each prepare script is reused with its cap switched off and its writer redirected.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SCRIPTS = Path(__file__).resolve().parent
TASKS = ["wids2021", "uci_diabetes130", "uci_support2", "uci_cdc_diabetes",
         "physionet_2012", "physionet_2019", "nhanes_htn", "metabric", "mexico_covid", "dryad_er_cbc"]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def saver(report):
    def write_bundle(name, frame, spec, summary, retrieval):
        path = DATASETS / name / "patients_full.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        report[name] = {"rows": len(frame), "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{name}: {len(frame)} rows -> {path.name}", flush=True)
    return write_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=TASKS)
    wanted = set(parser.parse_args().only or TASKS)
    report = {}

    if "wids2021" in wanted:
        wids = load_module("prepare_wids2021")
        wids.N_SAMPLE = None
        wids.main()
        path = DATASETS / "wids2021" / "patients_full.csv"
        report["wids2021"] = {"rows": sum(1 for _ in path.open()) - 1, "path": str(path.relative_to(ROOT)),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    ext1 = load_module("prepare_clinical_extension")
    ext3 = load_module("prepare_clinical_extension3")
    public = load_module("prepare_public_extension")
    for module in (ext1, ext3, public):
        module.write_bundle = saver(report)
    ext1.DIABETES_N = None
    ext3.SUPPORT2_N = ext3.CDC_N = None
    public.CAP = public.MEXICO_ROWS = None

    if "uci_diabetes130" in wanted:
        ext1.prepare_diabetes130(DATASETS / "uci_diabetes130/raw/data.csv", [])
    if "uci_support2" in wanted:
        ext3.prepare_support2(DATASETS / "uci_support2/raw/data.csv", [])
    if "uci_cdc_diabetes" in wanted:
        ext3.prepare_cdc_diabetes(DATASETS / "uci_cdc_diabetes/raw/data.csv", [])

    raw = lambda name: DATASETS / name / "raw"
    if "physionet_2012" in wanted:
        public.prepare_physionet_2012(raw("physionet_2012"), [])
    if "physionet_2019" in wanted:
        retrieval = public.download_ch2019(raw("physionet_2019"), public.CH2019_FULL_N)
        public.prepare_physionet_2019(raw("physionet_2019"), retrieval)
    if "nhanes_htn" in wanted:
        public.prepare_nhanes_htn(raw("nhanes_htn"), [])
    if "metabric" in wanted:
        public.prepare_metabric(raw("metabric"), [])
    if "mexico_covid" in wanted:
        public.prepare_mexico_covid(raw("mexico_covid"), [])
    if "dryad_er_cbc" in wanted:
        public.prepare_dryad_er_cbc(raw("dryad_er_cbc") / "DryadDB_bmjopen_2023_071937_R1_csv.csv", [])

    out = ROOT / "outputs" / "full_tables_preparation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(out.read_text())["tables"] if out.exists() else {}
    previous.update(report)
    out.write_text(json.dumps({"prepared_at": datetime.now(timezone.utc).isoformat(), "tables": previous},
                              indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
