#!/usr/bin/env python3
"""Verify generated scientific artifacts, not only isolated numerical functions."""
from pathlib import Path
import sys
import re
import json
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_scientific_revision import OUT, BACKBONES, clean
from jevtest.config import digest
from jevtest.revision_paths import PREVIOUS


def main():
    checks=[]
    def check(name,condition,detail):
        checks.append(dict(check=name,passed=bool(condition),detail=detail))
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    inputs={p:h for p,h in manifest['input_sha256'].items() if p.startswith(('outputs/','outputs_qwen3.5_','datasets/'))}
    changed=[p for p,h in inputs.items() if digest(ROOT/p)!=h]
    check('original_input_hashes',not changed,dict(files=len(inputs),changed=changed))
    original=pd.read_csv(OUT/'original_metrics.csv');baseline=pd.read_csv(PREVIOUS/'baseline/recomputed/ladder_table.csv')  # frozen v1 replay of the original pipeline
    names=dict(zip(BACKBONES,['0.8B','2B','4B','9B','27B','397B-A17B FP8']))
    differences=[]
    for b,oldname in names.items():
        new=original[original.backbone==b].merge(baseline,on=['dataset','outcome'])
        for metric,oldmetric in [('auroc','auc'),('balanced_accuracy','bacc')]:
            differences.append(float(np.abs(new[metric]-new[f'{oldmetric}_{oldname}']).max()))
    check('original_primary_point_estimates',max(differences)<1e-12,dict(max_difference=max(differences),comparisons=540))
    selected=pd.read_csv(OUT/'selected_markers.csv');cv=pd.read_csv(OUT/'marker_cv_candidates.csv');selection_errors=[]
    for row in selected.itertuples():
        candidates=cv[(cv.dataset==row.dataset)&(cv.outcome==row.outcome)]
        means=candidates.groupby('predictor').cv_auroc.mean().sort_values(ascending=False,kind='stable')
        if means.index[0]!=row.predictor:selection_errors.append(row.outcome)
    check('marker_selection_replays_from_development_cv',not selection_errors,dict(tasks=len(selected),errors=selection_errors))
    direct=pd.read_csv(OUT/'reference_pair_comparisons.csv')
    check('marker_versus_direction_comparisons',len(direct)==45 and np.isfinite(direct[['delta_auroc','delta_ba','auroc_lo','auroc_hi']]).all().all(),dict(tasks=len(direct)))
    label_errors=[];fits=0
    for path in (OUT/'calibration_parameters').glob('*/*/*.json'):
        dataset,outcome=path.parent.name,path.stem
        split=json.loads((ROOT/'outputs'/dataset/outcome/'holdout/split.json').read_text())
        for fit in json.loads(path.read_text()):
            fits+=1;ids=set(fit['calibration_ids'])
            if not ids.issubset(set(split['development'])) or ids&set(split['holdout']):label_errors.append(str(path))
    check('calibration_patient_separation',not label_errors,dict(fitted_conditions=fits,errors=label_errors))
    calibration=pd.read_csv(OUT/'calibration_metrics.csv');temp=calibration[calibration.method=='temperature']
    check('temperature_argmax_negative_control',bool(len(temp)) and np.all(np.abs(temp.delta_ba)<1e-12),dict(evaluated_conditions=len(temp),max_ba_change=float(temp.delta_ba.abs().max())))
    primary_invalid=original.groupby(['dataset','outcome']).invalid_missing_class_draws.first()
    check('missing_class_bootstrap_is_reported',bool((primary_invalid>0).any()),dict(tasks_with_invalid_draws=int((primary_invalid>0).sum()),draws_per_backbone=int(primary_invalid.sum())))
    md=(ROOT/'manuscript/npj_digital_medicine/parts/brief_communication.md').read_text()
    claimdoc=json.loads((OUT/'manuscript_claims.json').read_text());bad=[]
    claim_documents={name:(ROOT/'manuscript/npj_digital_medicine/parts'/name).read_text() for name in ('brief_communication.md','supplementary.md')}
    for entry in claimdoc['claims']:
        if digest(OUT/entry['source'])!=entry['source_sha256'] or entry['rendered'] not in claim_documents[entry.get('document','brief_communication.md')]:bad.append(entry['claim'])
    check('principal_numerical_claim_traceability',not bad,dict(claim_entries=len(claimdoc['claims']),errors=bad))
    wc=claimdoc['word_counts'];check('official_brief_communication_lengths',wc['title']<=15 and wc['abstract']<=70 and 1000<=wc['main_text']<=1500,wc)
    supplement=(ROOT/'manuscript/npj_digital_medicine/parts/supplementary.md').read_text()
    legends=[p for text in (md,supplement) for p in text.split('\n\n') if p.startswith(('**Fig.','**Supplementary Fig.'))]
    lengths=[len(re.findall(r'\S+',p)) for p in legends]
    check('figure_legend_lengths',len(legends)==12 and max(lengths)<=350,dict(legends=len(legends),maximum_words=max(lengths)))
    figures=list((OUT/'figures').glob('*.pdf'))
    check('complete_revised_figure_set',len(figures)==12 and all(p.stat().st_size>1000 for p in figures),dict(pdf_figures=len(figures)))
    tex=(ROOT/'manuscript/npj_digital_medicine/overleaf/manuscript.tex').read_text()
    labels=set(re.findall(r'\\label\{([^}]+)\}',tex));refs=set(re.findall(r'\\ref\{([^}]+)\}',tex))
    check('main_latex_figure_references_resolve',bool(refs) and refs.issubset(labels),dict(labels=sorted(labels),references=sorted(refs)))
    problems=[]
    for path in (OUT/'logs').glob('latex_*.log'):
        for line in path.read_text().splitlines():
            if line.startswith('! ') or 'undefined' in line.lower() or 'Overfull' in line or 'Missing character' in line:problems.append(dict(file=path.name,line=line))
    check('latex_errors_and_content_overflow',not problems,dict(problems=problems))
    with zipfile.ZipFile(ROOT/'manuscript/npj_digital_medicine/overleaf_bundle.zip') as bundle:
        forbidden=[name for name in bundle.namelist() if not name.endswith(('.tex','.pdf','latexmkrc'))]
    check('manuscript_bundle_contains_only_text_and_aggregate_figures',not forbidden,dict(unexpected_files=forbidden))
    report=dict(passed=all(c['passed'] for c in checks),checks=checks)
    (OUT/'artifact_qa.json').write_text(json.dumps(clean(report),indent=2))
    for item in checks:print(('PASS' if item['passed'] else 'FAIL')+' '+item['check'])
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
