#!/usr/bin/env python3
"""Replay preserved, intentionally uncorrected original analysis into a new directory.

This is solely for baseline provenance; never use its leaky reference results in
the revised manuscript. Original predictions and split assignments are read-only.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
SNAPSHOT=ROOT/'reports/scientific_revision_2026-10-25_v1/source_snapshot'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='new output directory; must not already exist')
    args=parser.parse_args()
    protected=ROOT/'pod_h200_run_2026-10-25_prior_none'
    if args.out.resolve()==protected or protected in args.out.resolve().parents:
        raise ValueError('the prior-none pod directory is protected')
    args.out.mkdir(parents=True,exist_ok=False)
    for filename in ('task_mechanism_covariates.py','plot_backbone_ladder.py'):
        spec=importlib.util.spec_from_file_location('original_'+Path(filename).stem,SNAPSHOT/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=ROOT;module.BASE=ROOT/'outputs';module.OUT=args.out
        if filename.startswith('plot_'):module.main([])
        else:module.main()


if __name__=='__main__':main()
