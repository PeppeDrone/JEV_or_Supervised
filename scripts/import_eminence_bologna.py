"""Create the pseudonymised Bologna external-validation cohort.

Only investigator-approved mappings are made. Fields without a Bologna
counterpart remain missing so the transported Careggi model encounters the
availability pattern of real-world external deployment.

Usage: .venv/bin/python scripts/import_eminence_bologna.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from bologna_working_diagnosis import code as working_diagnosis


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "datasets/diagnosis_ICUEEG/DATABASE EMINENCE Bologna 2024 Completo.xlsx"
DESTINATION = ROOT / "datasets/eminence_bologna_external"
WITH_WDX = ROOT / "datasets/eminence_bologna_external_wdx"
POSITIVE = {"CRISI EPILETTICA", "STATO EPILETTICO"}


def text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).upper()).strip()


def binary(value):
    """Yes/no, including the sheet's abbreviations and annotated affirmatives.

    N is no. SI? matches the existing SI ? rule. A SI note is yes unless it
    expresses doubt, a negation, or a previous rather than index event.
    """
    raw_text = text(value)
    if not raw_text:
        return pd.NA
    compact = raw_text.replace(" ", "")
    if compact in {"SI", "S", "SI?", "YES", "SI*"}:
        return 1
    if compact in {"NO", "N"}:
        return 0
    if "?" in raw_text or "PRECEDENT" in raw_text or raw_text.startswith("SI MA"):
        return pd.NA
    if re.search(r"\bNO\b", raw_text):
        return pd.NA
    if raw_text.startswith("SI ") or raw_text.startswith("SI("):
        return 1
    return pd.NA


def contains(value, *patterns):
    value = text(value)
    if not value or value == "NO":
        return 0 if value == "NO" else pd.NA
    return int(any(pattern in value for pattern in patterns))


# Careggi symptom_regression_ed: 0 none, 1 partial, 2 complete, 3 regressed then recurred.
def regression(value):
    value = text(value).replace(" ", "")
    if value in {"NO", "N"}:
        return 0
    if value == "PARZIALE":
        return 1
    if value in {"SI", "S"} or (value.startswith("SI(") and "RECIDIV" not in value):
        return 2
    if "RECIDIV" in value and value.startswith("SI"):
        return 3
    return pd.NA


def other_consult_code(value):
    value = text(value)
    if not value:
        return pd.NA
    if value in {"NO", "N"}:
        return 0
    return 1


def known_lesion(value):
    coded = binary(value)
    if pd.notna(coded):
        return coded
    value = text(value)
    if not value or value.startswith("NO"):
        return pd.NA
    return 1


_MINUTES = re.compile(r"^(\d+(?:\.\d+)?)\s*(MIN|MINUTI|ORE|ORA|H|GG|GIORNO|GIORNI|SEC|SECONDI)?$")
_UNIT_TO_MINUTES = {
    "MIN": 1, "MINUTI": 1, "ORE": 60, "ORA": 60, "H": 60,
    "GG": 1440, "GIORNO": 1440, "GIORNI": 1440, "SEC": 1 / 60, "SECONDI": 1 / 60,
}


def minutes(value):
    """Point durations. Ranges, inequalities, and unknown phrases stay missing."""
    raw_text = text(value).replace(",", ".")
    if not raw_text or raw_text in {"NON NOTO", "NO NOTO", "NN", "ORE", "MINUTI"}:
        return pd.NA
    match = _MINUTES.match(raw_text)
    if not match:
        return pd.NA
    return float(match.group(1)) * _UNIT_TO_MINUTES[match.group(2) or "MIN"]


def minor_focal(value):
    value = text(value)
    if not value:
        return pd.NA
    if value == "NO":
        return 0
    if "FOCAL" in value:
        return 1
    if "BILAT" in value or "DIFFUSE" in value:
        return 0
    return pd.NA


def indicator_from_grade(code, positive):
    if pd.isna(code):
        return pd.NA
    code = int(code)
    if code in positive:
        return 1
    if code in {0, 1, 2, 3}:
        return 0
    return pd.NA


def main():
    raw = pd.read_excel(SOURCE)
    source_spec = yaml.safe_load((ROOT / "datasets/diagnosis_ICUEEG/dataset.yaml").read_text())
    predictors = [value["column"] for value in source_spec["predictors"].values()]
    frame = pd.DataFrame(pd.NA, index=raw.index, columns=predictors)
    frame.insert(0, "study_id", [f"B{index + 1:04d}" for index in raw.index])

    direct = {
        "sex_male": "Genere", "care_pathway": "PDC",
        "sx_language_disturbance": "DIST DEL LINGUAGGIO", "sx_confusion": "CONFUSIONE",
        "sx_altered_consciousness": "ALTERATO STATO DI COSCIENZA", "sx_absence": "ASSENZA",
        "sx_psychomotor_slowing": "RALLENTAM IDEOMOTORIO", "sx_sensory_disturbance": "DIST SENSITIVI",
        "sx_weakness": "IPOSTENIA", "sx_headache": "CEFALEA", "sx_asthenia": "ASTENIA",
        "sx_agitation": "AGITAZIONE", "sx_motor_manifestations": "MANIFESTAZ MOTORIE",
        "sx_facial_deficit": "DEFICIT FACIALE", "sx_other_cranial_nerves": "altri nervi cranici ",
        "sx_amnesia": "AMNESIA", "sx_hallucinations": "ALLUCINAZ", "sx_catatonic_state": "STATO CATATONICO",
        "sx_vertigo": "VERTIGINI", "sx_visual_disturbance": "DIST VISIVO ", "sx_vomiting": "VOMITO",
        "sx_tongue_bite": "morsus", "sx_postural_instability": "INSTABIL POSTURALE",
        "sx_meningeal_signs": "segni meningei", "symptoms_during_eeg": "Sintomi durante esecuzione eeg",
        "witnessed_by_staff": "Episodio critico testimoniato da sanitari",
        "witnessed_by_others": "Episodio critico  testimoniato da altri",
        "known_epilepsy_idiopathic": "Epilettico noto idiopatico",
        "known_epilepsy_secondary": "Epilettico noto da causa secondaria", "fever": "Febbre",
        "electrolyte_alterations": "Alterazioni elettrolitiche ", "chemo_radiotherapy": "Chemio/Radioterapia",
        "previous_intracranial_event": "Pregresso int nch", "poor_baseline_conditions": "Scadenti condizioni di base",
        "nursing_home_resident": "Degente in RSA", "pregnancy_postpartum": "Gravidanza/Post-partum",
        "cardiac_comorbidities": "Problematiche cardiologiche", "hypertension": "Ipertensione",
        "diabetes": "Diabete", "dyslipidemia": "Dislipidemia", "thyroid_disorder": "Distiroidismo",
        "metabolic_alterations": "Alterazioni metaboliche",
        "sepsis": "Stato settico", "intoxication_any": "Intossicazione",
        "substance_misuse_history_any": "Storia di abuso di sostanze", "on_asm_therapy": "In th con AED",
        "asm_underdosed": "TH AED sottodosata", "antidepressant_antipsychotic_any": "Antidepressivi/Antipsicotici",
        "neurology_requested": "Richiesta neuro",
        "ecg_done": "ECG", "brain_ct_done": "TC encefalo", "brain_mri_done": "RMN encefalo",
        "neurology_consult": "Consul Neurologica ", "neurosurgery_consult": "Consul NCH",
        "anesthesiology_consult": "Consul Rianimat",
        "pharmacological_test": "Prova Farmacologica", "eeg_working_hours": "EEG in orario lavorativo attivo ",
        "eeg_on_call": "EEG in reperibilità", "eeg_epileptiform": "Anomalie epilettiche",
        "eeg_reduced_vigilance_phases": "Fasi di riduzione vigilanza", "eeg_triphasic_waves": "Trifasiche",
        "eeg_periodic_patterns": "Periodismi", "motor_seizure": "Crisi Motoria",
        "non_convulsive_seizure": "Crisi non convuls", "motor_status_epilepticus": "SE motorio",
        "non_convulsive_status_epilepticus": "SE non convuls",
        "motor_events_not_eeg_related": "Manifestaz motorie non correlate ad alterazioni EEG",
    }
    for target, source in direct.items():
        frame[target] = raw[source].map(binary)

    frame["sex_male"] = raw["Genere"].map(lambda value: {"M": 1, "F": 0}.get(text(value), pd.NA))
    frame["age_years"] = pd.to_numeric(raw["Età"], errors="coerce")
    frame["admission_year"] = pd.to_datetime(raw["Data accesso PS"], errors="coerce").dt.year
    frame["symptom_regression_ed"] = raw["regressione sintomatologia di accesso in DEAS"].map(regression)
    frame["other_consult"] = raw["Altra Consulenza"].map(other_consult_code)
    frame["known_brain_lesion"] = raw["Lesioni encefaliche note "].map(known_lesion)
    frame["recent_therapy_change_any"] = raw["Inserimento/Sospensione  recente  TH non AED"].map(binary)
    named_change = raw["Inserimento/Sospensione  recente  TH non AED"].map(text).isin({"BDZ", "BDZ E ALCOL"})
    frame.loc[named_change, "recent_therapy_change_any"] = 1
    frame["medication_before_eeg_any"] = raw["Farmaci Eseguiti prima di fare EEG"].map(binary)
    before_eeg = raw["Farmaci Eseguiti prima di fare EEG"].map(text).str.startswith("PRIMA DELL")
    frame.loc[before_eeg, "medication_before_eeg_any"] = 1
    frame["symptom_duration_min"] = raw["Durata sintomi (min)"].map(minutes)
    frame["admission_to_eeg_report_min"] = raw["Time accesso-referto EEG (min)"].map(minutes)
    frame["request_to_eeg_min"] = raw["Tempo da Richiesta Minuti"].map(minutes)

    neurological = raw["Problematiche neurologiche "]
    frame["neuro_comorbidity_any"] = neurological.map(lambda value: 0 if text(value) == "NO" else
                                                       (pd.NA if not text(value) else 1))
    neurological_rules = {
        "neuro_comorbidity_cerebrovascular": ("ISCHEM", "TIA", "ICH", "ESA", "ANEUR", "TROMB", "VASCOL", "INFART", "ANGIOPAT"),
        "neuro_comorbidity_tumour": ("K CEREBR", "MENINGIOM", "METAST", "NEOFORM", "TUMOR"),
        "neuro_comorbidity_cognitive_decline": ("COGNIT", "ALZHEIM", "NDD"),
        "neuro_comorbidity_hydrocephalus_or_shunt": ("IDROCEF", "DVP", "SHUNT"),
        "neuro_comorbidity_neurodevelopmental": ("AUTIST", "DOWN", "NEUROSVIL"),
        "neuro_comorbidity_headache_disorder": ("EMICRAN", "CEFALEA"),
        "neuro_comorbidity_head_injury": ("TRAUMA", "SUBDURAL", "EPIDURAL"),
        "neuro_comorbidity_parkinsonism": ("PARKIN",),
        "neuro_comorbidity_demyelinating": ("SCLEROSI MULT", "DEMIEL"),
        "neuro_comorbidity_toxic_metabolic_or_anoxic_encephalopathy": ("ANOSS", "IPOSS", "METABOL", "ENCEFALOPAT"),
    }
    for target, patterns in neurological_rules.items():
        frame[target] = neurological.map(lambda value: contains(value, *patterns))
    specific = frame[list(neurological_rules)].fillna(0).astype(int).sum(axis=1).gt(0)
    frame["neuro_comorbidity_other"] = np.where(frame.neuro_comorbidity_any.isna(), pd.NA,
                                                  ((frame.neuro_comorbidity_any == 1) & ~specific).astype(int))

    lesions = raw["Lesioni recenti Tc encefalo"]
    frame["ct_new_lesion_any"] = lesions.map(lambda value: 0 if text(value) == "NO" else
                                             (pd.NA if not text(value) else 1))
    lesion_rules = {
        "ct_new_lesion_ischaemia_or_vessel_occlusion": ("ISCHEM", "OCCLUS", "TROMB"),
        "ct_new_lesion_haemorrhage_or_contusion": ("ICH", "ESA", "EMATOM", "SANGUIN", "RISANGUIN", "SUBDURAL", "CONTUS"),
        "ct_new_lesion_tumour": ("K CEREBR", "MENINGIOM", "NEOFORMAT", "TUMOR", "GLIALE"),
        "ct_new_lesion_oedema_or_hydrocephalus": ("EDEMA", "IDROCEF"),
        "ct_new_lesion_vascular_malformation": ("MAV", "MALFORMA", "ANEUR"),
    }
    for target, patterns in lesion_rules.items():
        frame[target] = lesions.map(lambda value: contains(value, *patterns))
    lesion_text = lesions.map(text)
    frame["ct_new_lesion_lesion_unspecified"] = lesion_text.map(lambda value: 1 if value in {"SI", "S", "SI ?"} else
                                                                  (0 if value == "NO" else pd.NA))
    named_lesion = frame[list(lesion_rules)].fillna(0).astype(int).sum(axis=1).gt(0)
    frame["ct_new_lesion_other"] = np.where(frame.ct_new_lesion_any.isna(), pd.NA,
                                             ((frame.ct_new_lesion_any == 1) & ~named_lesion
                                              & frame.ct_new_lesion_lesion_unspecified.isna()).astype(int))

    frequency = raw["EEG frequenza"].map(text)
    frame["eeg_frequency"] = frequency.map({"NORMALE": 0, "RALLENTATO": 1, "RAPIDO (FARMACOL)": 0})
    frame["eeg_frequency_grade"] = frequency.map({"NORMALE": 0, "RALLENTATO": 2, "RAPIDO (FARMACOL)": 3})
    frame["eeg_frequency_pharmacological"] = frequency.map(
        lambda value: 1 if value == "RAPIDO (FARMACOL)" else (0 if value in {"NORMALE", "RALLENTATO"} else pd.NA))
    topography = raw["Topografia"].map(text)
    frame["eeg_topography"] = topography.map(
        lambda value: 1 if value in {"SCARSA", "NO"} else (0 if value in {"SI", "S"} else pd.NA))
    frame["eeg_topography_grade"] = topography.map({"SI": 0, "S": 0, "SCARSA": 2, "NO": 3})
    reactivity = raw["Reattività"].map(text)
    frame["eeg_reactivity"] = reactivity.map(
        lambda value: 1 if value in {"PARZIALE", "SCARSA", "NO"} else (0 if value in {"SI", "S"} else pd.NA))
    frame["eeg_reactivity_grade"] = reactivity.map({"SI": 0, "S": 0, "PARZIALE": 1, "SCARSA": 2, "NO": 3})
    asymmetry = raw["Asimmetria"].map(text)
    frame["eeg_asymmetry"] = asymmetry.map(lambda value: 1 if value == "SI" else (0 if value == "NO" else pd.NA))
    frame["eeg_asymmetry_grade"] = asymmetry.map(lambda value: 1 if value == "SI" else (0 if value == "NO" else pd.NA))
    major, minor = raw["Anomalie lente maggiori "], raw["Anomalie lente minori significative"]
    frame["eeg_major_slowing_type"] = major.map(
        lambda value: 0 if text(value) == "NO" else (3 if "FOCALI" in text(value) and "BILAT" in text(value)
        else 2 if ("BILAT" in text(value) or "DIFFUSE" in text(value)) else 1 if "FOCAL" in text(value)
        else 4 if text(value) else pd.NA))
    # Careggi defines these from major-slowing type: focal is type 1 or 3,
    # bilateral is type 2 or 3, and type 4 (present, location not stated) is missing.
    frame["eeg_focal_slowing"] = frame["eeg_major_slowing_type"].map(lambda code: indicator_from_grade(code, {1, 3}))
    frame["eeg_bilateral_slowing"] = frame["eeg_major_slowing_type"].map(lambda code: indicator_from_grade(code, {2, 3}))
    frame["eeg_minor_focal_slowing"] = minor.map(minor_focal)
    negative_eeg = raw["EEG NEGATIVI"].map(binary)
    frame["eeg_abnormal"] = negative_eeg.map(lambda value: 1 - value if pd.notna(value) else pd.NA)

    outcome = raw["DIAGNOSI UNICA"].map(text)
    frame["final_dx_epileptic"] = outcome.map(lambda value: 1 if value in POSITIVE else (pd.NA if not value else 0))

    DESTINATION.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DESTINATION / "patients.csv", index=False)
    external_spec = source_spec.copy()
    external_spec.update({
        "name": "eminence_bologna_external",
        "description": ("External Bologna cohort from DATABASE EMINENCE Bologna 2024 Completo.xlsx; "
                        "all available 2024 and 2026 records, predictors harmonised to the Careggi schema. "
                        "Unavailable fields remain missing."),
        "file": "patients.csv", "id_column": "study_id", "pseudonymize": False,
        # Keep the model-facing template identical to development; site identity
        # belongs in manuscript reporting, not in the patient prompt.
        "setting": source_spec["setting"],
        "observations_header": source_spec["observations_header"],
    })
    external_spec["outcomes"] = {
        "epileptic_final_diagnosis": {
            "column": "final_dx_epileptic",
            "classes": ["Non-epileptic diagnosis", "Epileptic seizure or status epilepticus"],
            "mapping": {"0": "Non-epileptic diagnosis", "1": "Epileptic seizure or status epilepticus"},
            "question": source_spec["outcomes"]["epileptic_final_diagnosis"]["question"],
        }
    }
    (DESTINATION / "dataset.yaml").write_text(yaml.safe_dump(external_spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
    notes = """# Bologna external cohort import

Source: DATABASE EMINENCE Bologna 2024 Completo.xlsx, sheet Foglio1.

Rows: 398 source records. DIAGNOSI UNICA is the outcome; CRISI EPILETTICA and
STATO EPILETTICO map to the positive class. Three records without this field
remain without outcome and are excluded by the analysis loader.

All 2024 and 2026 records are retained as one cohort. Unsupported Careggi
predictors are blank, never imputed. The outcome question is copied from the
Careggi specification. The workbook does not state a 48-hour endpoint.

Coding repaired against the Careggi value labels:
- Symptom regression: NO/N = 0, PARZIALE = 1, SI/S = 2, SI with RECIDIVA = 3.
  One cell, SI (CRISI IN TRIAGE), is coded 2 because it affirms regression and
  does not say recurrence.
- Other consultation: NO = 0; a named specialty = 1.
- Yes/no abbreviations: N = 0, SI? = 1 (same as SI ?), SI* and a SI note = 1.
  Doubt, negation (SI MA), and PRECEDENTE stay missing.
- Known brain lesion: a named lesion = 1. A NO followed by an exception stays missing.
- Recent non-ASM change: BDZ and BDZ E ALCOL = 1.
- Medication before EEG: a value that states it was before the EEG = 1.
- Durations: a bare number is minutes; MIN, ORE/ORA/H, GG/GIORNI, and SEC convert
  to minutes. Ranges, inequalities, and NON NOTO stay missing.
- Focal and bilateral slowing follow major-slowing type, as in Careggi.
  Minor focal slowing is read only from the minor-slowing column. Affirmative
  text that does not state focal, bilateral, or diffuse topography stays missing.

Neurological-history and CT-lesion subtype keywords are unchanged and still
need clinical review. Working diagnosis, previous-episode symptoms, intoxication
subtypes, pre-EEG drug class, sepsis source, and previous chemotherapy remain blank.
"""
    (DESTINATION / "DATASET_NOTES.md").write_text(notes, encoding="utf-8")

    # Variant with the admission working diagnosis: Bologna's structured codes where present,
    # otherwise the keyword dictionary on the free-text suspected diagnosis.
    coded = working_diagnosis(raw)
    variant = frame.copy()
    for column in coded.columns.drop("adm_dx_source"):
        variant[column] = coded[column]
    WITH_WDX.mkdir(parents=True, exist_ok=True)
    variant.to_csv(WITH_WDX / "patients.csv", index=False)
    pd.DataFrame({"study_id": frame.study_id, "adm_dx_source": coded.adm_dx_source}).to_csv(
        WITH_WDX / "working_dx_source.csv", index=False)
    external_spec["name"] = WITH_WDX.name
    external_spec["description"] = external_spec["description"].replace(
        "Unavailable fields remain missing.",
        "Admission working diagnosis coded from Bologna records; other unavailable fields remain missing.")
    (WITH_WDX / "dataset.yaml").write_text(yaml.safe_dump(external_spec, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")
    sources = coded.adm_dx_source.value_counts().to_dict()
    (WITH_WDX / "DATASET_NOTES.md").write_text(notes.replace(
        "Working diagnosis, previous-episode symptoms", "Previous-episode symptoms") + f"""
Admission working diagnosis (this variant only), by source: {sources}.
- Structured: DIAGNOSI ACCESSO 2 and DIAGNOSI sommata, which use the Careggi category names.
- Free text: keyword dictionary on "sospetto diagnostico di accesso in DEAS"
  (scripts/bologna_working_diagnosis.py). Where both exist (109 patients) the dictionary
  matched the structured unified category in 44; it is not clinically validated.
- The unified category follows the rule observed in the Careggi coding.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
