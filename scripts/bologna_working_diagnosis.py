"""Admission working diagnosis for Bologna, coded to the Careggi categories.

Source precedence: Bologna's own structured codes (DIAGNOSI ACCESSO 2, DIAGNOSI sommata), which
use the Careggi category names, then a keyword dictionary on the free-text suspected diagnosis
(sospetto diagnostico di accesso in DEAS). The unified category follows the rule observed in Careggi.

Usage (review only, writes nothing used by the models):
    .venv/bin/python scripts/bologna_working_diagnosis.py
writes datasets/eminence_bologna_external/working_dx_review.csv and prints agreement of the
free-text dictionary with the structured codes where both exist.
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "datasets/diagnosis_ICUEEG/DATABASE EMINENCE Bologna 2024 Completo.xlsx"
FLAGS = ["seizure", "status_epilepticus", "absence", "transient_loss_of_consciousness", "stroke", "tia",
         "altered_consciousness", "head_injury", "fall_unknown_mechanism", "transient_global_amnesia",
         "encephalitis", "migraine_with_aura", "other"]

STRUCTURED = {
    "CRISI EPILETTICA": "seizure", "CRISI EPIELTTICA": "seizure", "STATO EPILETTICO": "status_epilepticus",
    "ASSENZA": "absence", "PDC": "transient_loss_of_consciousness", "PCD": "transient_loss_of_consciousness",
    "STROKE": "stroke", "TIA": "tia", "ALTERAZIONE STATO DI COSCIENZA": "altered_consciousness",
    "ALTERATO STATO DI COSCIENZA": "altered_consciousness", "TRAUMA CRANICO": "head_injury",
    "CADUTA A DINAMICA IGNOTA": "fall_unknown_mechanism", "CADUTA A DINAMICA INGOTA": "fall_unknown_mechanism",
    "CADUATA DINAMICA IGNOTA": "fall_unknown_mechanism", "ENCEFALITE": "encephalitis",
    "EMICRANIA CON AURA": "migraine_with_aura", "NESSUNA DIAGNOSI": None, "NESSUNADIAGNOSI": None,
}

# Free-text keywords (upper case, whitespace-normalised). Each needs clinical approval.
KEYWORDS = {
    "status_epilepticus": [r"STATO EPILETTICO", r"STATO DI MALE", r"SUBENTRANT"],
    "seizure": [r"\bCRISI\b(?! IPERTENSIVA)", r"\bCRICO\b", r"CONVULS", r"\bSCOSSE", r"\bSCOSSER\b", r"CLONI",
                r"TONICO[ -]?CLONIC", r"POST CRITICO", r"\bSPASMI\b"],
    "absence": [r"ASSENZA", r"PERDITA DI CONTATTO", r"NON CONTATTO", r"ARRESTO DEL CONTATTO"],
    "transient_loss_of_consciousness": [r"PERDITA DI COSCIENZA", r"PEDITA DI COSCIENZA", r"\bPDC\b", r"\bPCD\b",
                                        r"SINCOP", r"SICNOP", r"LIPOTIM"],
    "stroke": [r"STROKE", r"ICTUS", r"EMIPARES", r"EMIPAREIS", r"EMIPLEGIA(?! SN TRANSITORIA)", r"EMISINDROME",
               r"EMISINDDROME"],
    "tia": [r"\bTIA\b", r"EMIPLEGIA SN TRANSITORIA"],
    "altered_consciousness": [r"SOPOR", r"CONFUSION", r"CONFUSIONE", r"DISORIENTAMENTO", r"STATO DI COSCIENZA",
                              r"\bCOMA\b", r"RIDOTTA VIGILANZA", r"ASSOPIMENTO", r"ARESPONSIV", r"ARESPINSIV",
                              r"ARESPOSNIV", r"ALTERAIZONE DI COSCIENZA", r"STATO DI AGITAZION",
                              r"AGITAZIONE PSICOMOTORIA", r"CATATONIA"],
    "head_injury": [r"TRAUMA CRANI", r"TRAUMA FACCIALE", r"TRAUMATISMO FACIALE", r"POLITRAUMA"],
    "fall_unknown_mechanism": [r"CADUTA NON TESTIMONIATA", r"CADUTE NON TESTIMONIATE", r"DINAMICA NON NOTA",
                               r"CADUTA CON STAZIONAMENTO", r"STAZIONAMENTO A TERRA"],
    "transient_global_amnesia": [r"AMNESIA GLOBALE TRANSITORIA", r"\bAMNESIA\b"],
    "encephalitis": [r"ENCEFALIT"],
    "migraine_with_aura": [r"EMICRANIA"],
}


def text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).upper()).strip()


def from_structured(*values):
    found, seen = set(), False
    for value in values:
        for part in filter(None, (p.strip() for p in text(value).split("/"))):
            seen = True
            if STRUCTURED.get(part):
                found.add(STRUCTURED[part])
    return found if seen else None


def from_text(value):
    value = text(value)
    if not value:
        return None
    return {flag for flag, patterns in KEYWORDS.items() if any(re.search(p, value) for p in patterns)}


def unified(found):
    """Careggi rule: 1 seizure, 2 suspected seizure, 3 vascular event with seizure, 4 stroke, 5 TIA,
    6 transient loss of consciousness, 7 altered consciousness, 8 status epilepticus, 9 TGA,
    10 head injury, 11 fall of unknown mechanism, 12 migraine with aura, 13 encephalitis, 14 none."""
    f = found
    if "status_epilepticus" in f:
        return 8
    if "seizure" in f:
        if f & {"stroke", "tia"}:
            return 3
        return 2 if f - {"seizure"} else 1
    if "absence" in f:
        return 3 if "stroke" in f else 2
    if "stroke" in f:
        if "altered_consciousness" in f:
            return 7
        return 5 if "tia" in f else 4
    if "tia" in f:
        return 9 if "transient_global_amnesia" in f else 5
    if "altered_consciousness" in f:
        return 7
    if "transient_loss_of_consciousness" in f:
        return 6
    if "transient_global_amnesia" in f:
        return 9
    if "encephalitis" in f:
        return 13
    if "head_injury" in f:
        return 10
    if "fall_unknown_mechanism" in f:
        return 11
    if "migraine_with_aura" in f:
        return 12
    return 14


def code(raw):
    """One row per patient: adm_dx_* flags, adm_dx_unified and the source used."""
    rows = []
    for _, row in raw.iterrows():
        structured = from_structured(row["DIAGNOSI ACCESSO 2"], row["DIAGNOSI sommata"])
        free = from_text(row["sospetto diagnostico di accesso in DEAS"])
        found, source = (structured, "structured") if structured is not None else (free, "free text")
        if found is None:
            rows.append({**{f"adm_dx_{f}": pd.NA for f in FLAGS}, "adm_dx_unified": pd.NA, "adm_dx_source": "missing"})
            continue
        rows.append({**{f"adm_dx_{f}": int(f in found) for f in FLAGS}, "adm_dx_unified": unified(found),
                     "adm_dx_source": source})
    return pd.DataFrame(rows, index=raw.index)


def main():
    raw = pd.read_excel(SOURCE)
    coded = code(raw)
    free = raw["sospetto diagnostico di accesso in DEAS"].map(from_text)
    structured = [from_structured(a, b) for a, b in zip(raw["DIAGNOSI ACCESSO 2"], raw["DIAGNOSI sommata"])]
    both = [i for i, s in enumerate(structured) if s is not None and free.iloc[i] is not None]
    print(f"Sources: {coded.adm_dx_source.value_counts().to_dict()}")
    print(f"\nFree-text dictionary vs Bologna structured codes, {len(both)} patients with both:")
    exact = sum(free.iloc[i] == structured[i] for i in both)
    same_unified = sum(unified(free.iloc[i]) == unified(structured[i]) for i in both)
    print(f"  identical flag set {exact}/{len(both)}; identical unified category {same_unified}/{len(both)}")
    for flag in FLAGS[:-1]:
        s = [flag in structured[i] for i in both]
        t = [flag in free.iloc[i] for i in both]
        agree = sum(a == b for a, b in zip(s, t))
        print(f"  {flag:34} structured {sum(s):3}  text {sum(t):3}  agree {agree}/{len(both)}")
    review = pd.DataFrame({"free_text": raw["sospetto diagnostico di accesso in DEAS"].map(text),
                           "proposed_flags": free.map(lambda f: "; ".join(sorted(f)) if f else "(none)"),
                           "proposed_unified": free.map(lambda f: unified(f) if f is not None else None)})
    review = review.groupby(["free_text", "proposed_flags", "proposed_unified"], dropna=False).size() \
        .rename("patients").reset_index().sort_values("patients", ascending=False)
    path = ROOT / "datasets/eminence_bologna_external/working_dx_review.csv"
    review.to_csv(path, index=False)
    print(f"\nReview file: {path.relative_to(ROOT)} ({len(review)} distinct texts)")
    labels = {1: "Epileptic seizure", 2: "Suspected seizure", 3: "Vascular event with seizure", 4: "Stroke",
              5: "TIA", 6: "Transient loss of consciousness", 7: "Altered consciousness", 8: "Status epilepticus",
              9: "Transient global amnesia", 10: "Head injury", 11: "Fall of unknown mechanism",
              12: "Migraine with aura", 13: "Encephalitis", 14: "No working diagnosis"}
    print("\nResulting unified distribution (Bologna):")
    print(coded.adm_dx_unified.map(labels).value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
