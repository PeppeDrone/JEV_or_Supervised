# UCI core benchmark summary

Models: M2 (XGBoost), ZS_raw (System 1).
Same stratified 5-fold outer CV (seed 20261008) as the existing NeuroJev benchmark.
Metrics use the repository `jevtest.metrics` definitions; multiclass AUROC is macro one-vs-rest.
Bootstrap CIs are patient-level resamples of out-of-fold predictions (conditional on fitted folds).
Differences vs M2 (XGBoost) use paired resamples when both arms score the same patients and independent resamples when XGBoost was scored on the whole table; intervals that include zero are not treated as supported improvements.

## Per-dataset notes

### cleveland_heart / heart_disease
303 observations, 13 predictors, 2 classes (No angiographic coronary artery disease (less than 50% narrowing), Angiographic coronary artery disease (50% or more narrowing)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for cleveland_heart/heart_disease); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for cleveland_heart/heart_disease).


### uci_hcv / hcv_category
608 observations, 12 predictors, 4 classes (Healthy blood donor, Hepatitis C without fibrosis, Hepatitis C with fibrosis, Hepatitis C with cirrhosis).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_hcv/hcv_category); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_hcv/hcv_category).


### uci_mammographic_mass / malignancy
961 observations, 4 predictors, 2 classes (Benign, Malignant).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_mammographic_mass/malignancy); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_mammographic_mass/malignancy).


### uci_heart_failure / mortality
299 observations, 11 predictors, 2 classes (Alive at the end of follow-up, Died during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_heart_failure/mortality); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_heart_failure/mortality).


### uci_ilpd / liver_disease
583 observations, 10 predictors, 2 classes (Non-liver-patient group, Liver-patient group (liver disease)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_ilpd/liver_disease); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_ilpd/liver_disease).


### uci_diabetes130 / readmit_30d
1000 observations, 18 predictors, 2 classes (Not readmitted within 30 days, Readmitted within 30 days).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_diabetes130/readmit_30d); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_diabetes130/readmit_30d).


### uci_cervical / biopsy
858 observations, 29 predictors, 2 classes (Negative biopsy, Positive biopsy).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_cervical/biopsy); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_cervical/biopsy).


### wids2021 / diabetes_mellitus
1000 observations, 32 predictors, 2 classes (No diagnosis of diabetes mellitus, Pre-existing diagnosis of diabetes mellitus).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for wids2021/diabetes_mellitus); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for wids2021/diabetes_mellitus).


### uci_breast_coimbra / breast_cancer
116 observations, 9 predictors, 2 classes (Healthy control, Breast cancer).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_breast_coimbra/breast_cancer); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_breast_coimbra/breast_cancer).


### uci_hepatitis / died
155 observations, 19 predictors, 2 classes (Alive at end of follow-up, Died during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_hepatitis/died); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_hepatitis/died).


### uci_bone_marrow / bmt_died
187 observations, 35 predictors, 2 classes (Alive at end of follow-up, Died during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_bone_marrow/bmt_died); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_bone_marrow/bmt_died).


### uci_haberman / died_5y
306 observations, 3 predictors, 2 classes (Survived 5 years or longer after surgery, Died within 5 years of surgery).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_haberman/died_5y); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_haberman/died_5y).


### uci_maternal_risk / risk_level
1014 observations, 6 predictors, 3 classes (Low risk, Medium risk, High risk).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_maternal_risk/risk_level); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_maternal_risk/risk_level).


### uci_cirrhosis / cirrhosis_died
418 observations, 17 predictors, 2 classes (Alive or transplanted at end of follow-up, Died during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_cirrhosis/cirrhosis_died); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_cirrhosis/cirrhosis_died).


### uci_support2 / hosp_death
1000 observations, 37 predictors, 2 classes (Discharged alive, Died in hospital).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_support2/hosp_death); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_support2/hosp_death).


### uci_aids_cid / aids_event
2139 observations, 19 predictors, 2 classes (Censored without reaching the endpoint, Reached the primary endpoint (CD4 decline of 50% or more, AIDS-defining illness or death)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_aids_cid/aids_event); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_aids_cid/aids_event).


### uci_glioma / gbm
839 observations, 23 predictors, 2 classes (Lower-grade glioma (WHO grade II or III), Glioblastoma (WHO grade IV)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_glioma/gbm); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_glioma/gbm).


### uci_vertebral / pathology
310 observations, 6 predictors, 3 classes (Normal spine, Disc hernia, Spondylolisthesis).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_vertebral/pathology); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_vertebral/pathology).


### uci_breast_recurrence / recurrence
286 observations, 9 predictors, 2 classes (No recurrence during follow-up, Recurrence during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_breast_recurrence/recurrence); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_breast_recurrence/recurrence).


### uci_cdc_diabetes / cdc_diabetes
1000 observations, 21 predictors, 2 classes (No diabetes, Prediabetes or diabetes).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_cdc_diabetes/cdc_diabetes); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_cdc_diabetes/cdc_diabetes).


### uci_parkinsons / parkinsons
195 observations, 22 predictors, 2 classes (Healthy speaker, Speaker with Parkinson's disease).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_parkinsons/parkinsons); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_parkinsons/parkinsons).


### uci_spectf / abnormal
267 observations, 44 predictors, 2 classes (Normal perfusion study, Abnormal perfusion study).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_spectf/abnormal); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_spectf/abnormal).


### uci_dermatology / dermatological_diagnosis
366 observations, 34 predictors, 6 classes (Psoriasis, Seborrhoeic dermatitis, Lichen planus, Pityriasis rosea, Chronic dermatitis, Pityriasis rubra pilaris).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_dermatology/dermatological_diagnosis); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_dermatology/dermatological_diagnosis).


### uci_thyroid_recur / recurred
383 observations, 16 predictors, 2 classes (No recurrence during follow-up, Recurrence during follow-up).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_thyroid_recur/recurred); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_thyroid_recur/recurred).


### uci_ckd / ckd_status
400 observations, 24 predictors, 2 classes (No chronic kidney disease, Chronic kidney disease).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_ckd/ckd_status); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_ckd/ckd_status).


### uci_arrhythmia / arrhythmia_class
429 observations, 279 predictors, 8 classes (Normal ECG, Ischaemic changes (coronary artery disease), Old anterior myocardial infarction, Old inferior myocardial infarction, Sinus tachycardia, Sinus bradycardia, Right bundle branch block, Other, unclassified arrhythmia).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_arrhythmia/arrhythmia_class); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_arrhythmia/arrhythmia_class).


### uci_thoracic / died_1y
470 observations, 16 predictors, 2 classes (Alive one year after surgery, Died within one year of surgery).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_thoracic/died_1y); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_thoracic/died_1y).


### uci_early_diabetes / diabetes
520 observations, 16 predictors, 2 classes (Negative for diabetes, Positive for diabetes).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_early_diabetes/diabetes); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_early_diabetes/diabetes).


### uci_wdbc / diagnosis
569 observations, 30 predictors, 2 classes (Benign, Malignant).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_wdbc/diagnosis); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_wdbc/diagnosis).


### uci_wbc_original / malignant
699 observations, 9 predictors, 2 classes (Benign, Malignant).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `classical` stage for uci_wbc_original/malignant); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec); rerun the `zeroshot` stage for uci_wbc_original/malignant).


### uci_autism / asd
704 observations, 18 predictors, 2 classes (AQ-10 screen negative, AQ-10 screen positive (6 or more items endorsed)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_autism/asd); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_autism/asd).


### uci_retinopathy / retinopathy
1151 observations, 19 predictors, 2 classes (No signs of diabetic retinopathy, Signs of diabetic retinopathy).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_retinopathy/retinopathy); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_retinopathy/retinopathy).


### uci_mi_lethal / lethal
1700 observations, 111 predictors, 2 classes (Survived the hospital stay, Died during the hospital stay).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_mi_lethal/lethal); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_mi_lethal/lethal).


### uci_obesity / obesity_class
2111 observations, 16 predictors, 7 classes (Insufficient weight (BMI below 18.5), Normal weight (BMI 18.5-24.9), Overweight level I (BMI 25-27.4), Overweight level II (BMI 27.5-29.9), Obesity type I (BMI 30-34.9), Obesity type II (BMI 35-39.9), Obesity type III (BMI 40 or more)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_obesity/obesity_class); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_obesity/obesity_class).


### uci_cardiotocography / nsp
2126 observations, 21 predictors, 3 classes (Normal fetal state, Suspect fetal state, Pathologic fetal state).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_cardiotocography/nsp); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_cardiotocography/nsp).


### uci_thyroid / hypothyroid_status
3163 observations, 25 predictors, 2 classes (Not hypothyroid, Hypothyroid).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for uci_thyroid/hypothyroid_status); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for uci_thyroid/hypothyroid_status).


### ppd_bangladesh / ppd_high
800 observations, 40 predictors, 2 classes (Edinburgh Postnatal Depression Scale below 13, Edinburgh Postnatal Depression Scale 13 or higher (probable depression)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for ppd_bangladesh/ppd_high); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for ppd_bangladesh/ppd_high).


### hip_fracture_elos / extended_los
1137 observations, 18 predictors, 2 classes (Hospital stay shorter than 14 days, Extended hospital stay of 14 days or more).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for hip_fracture_elos/extended_los); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for hip_fracture_elos/extended_los).


### berlin_appendicitis / appendicitis
590 observations, 12 predictors, 2 classes (No appendicitis, Acute appendicitis confirmed on histopathology).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for berlin_appendicitis/appendicitis); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for berlin_appendicitis/appendicitis).


### physionet_2012 / icu_death
1000 observations, 76 predictors, 2 classes (Survived to hospital discharge, Died in hospital).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for physionet_2012/icu_death); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for physionet_2012/icu_death).


### physionet_2019 / sepsis
1000 observations, 38 predictors, 2 classes (No sepsis during the ICU stay, Sepsis (Sepsis-3 criteria) during the ICU stay).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for physionet_2019/sepsis); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for physionet_2019/sepsis).


### nhanes_htn / hypertension
1000 observations, 12 predictors, 2 classes (No hypertension, Hypertension (measured blood pressure 130/80 mmHg or higher, or on antihypertensive medication)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for nhanes_htn/hypertension); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for nhanes_htn/hypertension).


### metabric / pam50_subtype
1000 observations, 13 predictors, 6 classes (Basal-like, HER2-enriched, Luminal A, Luminal B, Normal-like, Claudin-low).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for metabric/pam50_subtype); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for metabric/pam50_subtype).


### mexico_covid / covid_death
1000 observations, 16 predictors, 2 classes (No death registered, Died).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for mexico_covid/covid_death); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for mexico_covid/covid_death).


### dryad_er_cbc / er_death
1000 observations, 16 predictors, 2 classes (Alive at the end of follow-up, Died from any cause during follow-up (up to about 17 months)).
Unavailable: M2 (M2 predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `classical` stage for dryad_er_cbc/er_death); ZS_raw (ZS_raw predictions rejected: stale provenance (data_hash, dataset_spec, classes); rerun the `zeroshot` stage for dryad_er_cbc/er_death).


## Binary vs multiclass

- Binary tasks (Cleveland, CKD, mammographic mass): AUROC is the positive-class ROC; AUPRC is reported.
- Multiclass tasks (dermatology, HCV): AUROC is macro one-vs-rest; AUPRC is left blank.
- Do not pool AUROC across unrelated datasets.

CSV: `outputs/uci_core_benchmark_summary.csv`
