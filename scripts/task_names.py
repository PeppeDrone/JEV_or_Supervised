"""Readable task names for display in figures and tables.

The repository identifies each task by its dataset directory (``uci_wbc_original``, ``dryad_er_cbc`` ...). Those
slugs stay in the CSVs; anything a reader sees goes through ``task_name`` so that manuscript figures and
supplementary tables carry the clinical task rather than the storage name.
"""

TASK_NAMES = {
    "cleveland_heart": "Cleveland heart disease",
    "uci_hcv": "Hepatitis C category",
    "uci_mammographic_mass": "Mammographic mass",
    "uci_heart_failure": "Heart failure mortality",
    "uci_ilpd": "Indian liver disease",
    "uci_diabetes130": "Diabetes 30-day readmission",
    "uci_cervical": "Cervical cancer biopsy",
    "wids2021": "ICU diabetes, WiDS 2021",
    "uci_breast_coimbra": "Breast cancer Coimbra",
    "uci_hepatitis": "Hepatitis mortality",
    "uci_bone_marrow": "Bone marrow transplant death",
    "uci_haberman": "Haberman 5-year survival",
    "uci_maternal_risk": "Maternal health risk",
    "uci_cirrhosis": "Cirrhosis survival",
    "uci_support2": "SUPPORT2 hospital death",
    "uci_aids_cid": "AIDS clinical trial event",
    "uci_glioma": "Glioma grade",
    "uci_vertebral": "Vertebral column pathology",
    "uci_breast_recurrence": "Breast cancer recurrence",
    "uci_cdc_diabetes": "CDC diabetes indicators",
    "uci_parkinsons": "Parkinson's disease, voice",
    "uci_spectf": "SPECT heart perfusion",
    "uci_dermatology": "Dermatology diagnosis",
    "uci_thyroid_recur": "Thyroid cancer recurrence",
    "uci_ckd": "Chronic kidney disease",
    "uci_arrhythmia": "Cardiac arrhythmia class",
    "uci_thoracic": "Thoracic surgery 1-year death",
    "uci_early_diabetes": "Early-stage diabetes",
    "uci_wdbc": "Breast cancer Wisconsin, diagnostic",
    "uci_wbc_original": "Breast cancer Wisconsin, original",
    "uci_autism": "Adult autism screening",
    "uci_retinopathy": "Diabetic retinopathy",
    "uci_mi_lethal": "Myocardial infarction death",
    "uci_obesity": "Obesity level",
    "uci_cardiotocography": "Cardiotocography",
    "uci_thyroid": "Hypothyroidism",
    "ppd_bangladesh": "Postpartum depression",
    "hip_fracture_elos": "Hip fracture length of stay",
    "berlin_appendicitis": "Paediatric appendicitis",
    "physionet_2012": "ICU death, PhysioNet 2012",
    "physionet_2019": "ICU sepsis, PhysioNet 2019",
    "nhanes_htn": "NHANES hypertension",
    "metabric": "METABRIC PAM50 subtype",
    "mexico_covid": "COVID-19 death, Mexico",
    "dryad_er_cbc": "Emergency-room death, blood count",
}


def task_name(dataset: str, fallback: str | None = None) -> str:
    """Readable name for a dataset slug; falls back to the slug with underscores spaced and a capital initial."""
    if dataset in TASK_NAMES:
        return TASK_NAMES[dataset]
    raw = fallback or dataset.replace("uci_", "").replace("_", " ")
    return raw[:1].upper() + raw[1:]
