"""Canonical dataset × outcome list for figures, analysis, and queues.

Every prepared bundle under ``datasets/*/dataset.yaml`` (except EXAMPLE_*) belongs
here. Do not subset this list for “clean” paper figures.
"""
from pathlib import Path

import yaml

# Existing 20 keep their current figure-row order; later pairs follow, small first, then the public extension.
ALL_PAIRS = [
    ("cleveland_heart", "heart_disease"),
    ("uci_hcv", "hcv_category"),
    ("uci_mammographic_mass", "malignancy"),
    ("uci_heart_failure", "mortality"),
    ("uci_ilpd", "liver_disease"),
    ("uci_diabetes130", "readmit_30d"),
    ("uci_cervical", "biopsy"),
    ("wids2021", "diabetes_mellitus"),
    ("uci_breast_coimbra", "breast_cancer"),
    ("uci_hepatitis", "died"),
    ("uci_bone_marrow", "bmt_died"),
    ("uci_haberman", "died_5y"),
    ("uci_maternal_risk", "risk_level"),
    ("uci_cirrhosis", "cirrhosis_died"),
    ("uci_support2", "hosp_death"),
    ("uci_aids_cid", "aids_event"),
    ("uci_glioma", "gbm"),
    ("uci_vertebral", "pathology"),
    ("uci_breast_recurrence", "recurrence"),
    ("uci_cdc_diabetes", "cdc_diabetes"),
    ("uci_parkinsons", "parkinsons"),
    ("uci_spectf", "abnormal"),
    ("uci_dermatology", "dermatological_diagnosis"),
    ("uci_thyroid_recur", "recurred"),
    ("uci_ckd", "ckd_status"),
    ("uci_arrhythmia", "arrhythmia_class"),
    ("uci_thoracic", "died_1y"),
    ("uci_early_diabetes", "diabetes"),
    ("uci_wdbc", "diagnosis"),
    ("uci_wbc_original", "malignant"),
    ("uci_autism", "asd"),
    ("uci_retinopathy", "retinopathy"),
    ("uci_mi_lethal", "lethal"),
    ("uci_obesity", "obesity_class"),
    ("uci_cardiotocography", "nsp"),
    ("uci_thyroid", "hypothyroid_status"),
    ("ppd_bangladesh", "ppd_high"),
    ("hip_fracture_elos", "extended_los"),
    ("berlin_appendicitis", "appendicitis"),
    ("physionet_2012", "icu_death"),
    ("physionet_2019", "sepsis"),
    ("nhanes_htn", "hypertension"),
    ("metabric", "pam50_subtype"),
    ("mexico_covid", "covid_death"),
    ("dryad_er_cbc", "er_death"),
]

OUTCOME_LABEL = {
    "heart_disease": "Cleveland heart",
    "ckd_status": "CKD",
    "dermatological_diagnosis": "Dermatology",
    "hcv_category": "HCV",
    "malignancy": "Mammography",
    "diagnosis": "WDBC breast",
    "hypothyroid_status": "Thyroid",
    "arrhythmia_class": "Arrhythmia",
    "mortality": "Heart failure",
    "liver_disease": "ILPD liver",
    "readmit_30d": "Diabetes130 readmit",
    "biopsy": "Cervical biopsy",
    "diabetes_mellitus": "WiDS2021 diabetes",
    "breast_cancer": "Coimbra breast",
    "diabetes": "Early diabetes",
    "abnormal": "SPECTF heart",
    "died": "Hepatitis survival",
    "obesity_class": "Obesity class",
    "bmt_died": "BMT survival",
    "parkinsons": "Parkinson",
    "died_5y": "Haberman survival",
    "lethal": "MI lethal",
    "risk_level": "Maternal risk",
    "recurred": "Thyroid recur",
    "cirrhosis_died": "Cirrhosis death",
    "hosp_death": "SUPPORT2 death",
    "aids_event": "AIDS event",
    "gbm": "Glioma GBM",
    "retinopathy": "Retinopathy",
    "nsp": "CTG fetal status",
    "died_1y": "Thoracic 1y death",
    "malignant": "WBC original",
    "pathology": "Vertebral path",
    "asd": "Autism screen",
    "recurrence": "Breast recur",
    "cdc_diabetes": "CDC diabetes",
    "ppd_high": "Bangladesh PPD",
    "extended_los": "Hip fracture eLOS",
    "appendicitis": "Berlin appendicitis",
    "icu_death": "Challenge 2012 death",
    "sepsis": "Challenge 2019 sepsis",
    "hypertension": "NHANES hypertension",
    "pam50_subtype": "METABRIC PAM50",
    "covid_death": "Mexico COVID death",
    "er_death": "ER CBC death",
}


def discover_pairs(root):
    """Every non-example ``datasets/*/dataset.yaml`` outcome."""
    pairs = []
    datasets = Path(root) / "datasets"
    for yml in sorted(datasets.glob("*/dataset.yaml")):
        if yml.parent.name.startswith("EXAMPLE"):
            continue
        data = yaml.safe_load(yml.read_text()) or {}
        for outcome in (data.get("outcomes") or {}):
            pairs.append((yml.parent.name, outcome))
    return pairs


def missing_from_yaml(root):
    """Pairs present in YAML but not in ``ALL_PAIRS`` (should be empty)."""
    return sorted(set(discover_pairs(root)) - set(ALL_PAIRS))
