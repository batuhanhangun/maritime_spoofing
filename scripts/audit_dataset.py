#!/usr/bin/env python
"""Single-feature AUC audit of a labeled feature table (simulator-fingerprint
detection).

For every feature, computes AUC as a standalone classifier on:
  * the full dataset and each scenario,
  * zero-intensity subsets (A3 shift_speed == 0; A1 distance == time == 0),
    where any separation must come from capture physics (inter-receiver
    baseline collapse) or from simulation artifacts.

Interpretation rules used for the journal paper:
  * AUC >= 0.99 on a FULL scenario for a feature whose physical mechanism
    cannot explain intensity-independent separation -> artifact suspect.
  * Separation at zero intensity is legitimate ONLY for capture-mechanism
    features (inter_rx_*). On MARSIM this audit flags the entire SNR family
    (mean_snr_median: AUC 1.0000 on A1 and A2; ~0.07 dB deterministic offset).

Usage:
    python scripts/audit_dataset.py --config config.yaml
    python scripts/audit_dataset.py --features data/marsim_features.csv --out results/audit
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mspoof.ablation import family_of  # noqa: E402

META = ['filename', 'scenario', 'label', 'index',
        'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value']


def audit(df, oracle_auc=0.99, zero_auc=0.75):
    y = (df['label'] == 'spoofed').astype(int).values
    feats = [c for c in df.columns if c not in META + ['_error']]
    subsets = {'overall': np.ones(len(df), bool)}
    for scen in sorted(df['scenario'].dropna().unique()):
        subsets[scen] = (df['scenario'] == scen).values
    m = ((df['scenario'] == 'A3') & (df['param_2_name'] == 'shift_speed')
         & (df['param_2_value'] == 0))
    if m.any():
        subsets['A3_speed0'] = m.values
    m = ((df['scenario'] == 'A1') & (df['param_1_value'] == 0)
         & (df['param_2_value'] == 0))
    if m.any():
        subsets['A1_zero'] = m.values

    rows = []
    for f in feats:
        x = pd.to_numeric(df[f], errors='coerce').values.astype(float)
        med = np.nanmedian(x) if not np.all(np.isnan(x)) else 0.0
        x = np.where(np.isnan(x), med, x)
        r = {'feature': f, 'family': family_of(f) or 'other'}
        for name, mask in subsets.items():
            ys, xs = y[mask], x[mask]
            if len(np.unique(ys)) < 2 or np.std(xs) == 0:
                r[name] = np.nan
            else:
                a = roc_auc_score(ys, xs)
                r[name] = max(a, 1 - a)
        rows.append(r)
    res = pd.DataFrame(rows).set_index('feature')

    scen_cols = [c for c in res.columns if c not in ('family', 'overall',
                                                     'A3_speed0', 'A1_zero')]
    res['oracle_any_scenario'] = (res[scen_cols] >= oracle_auc).any(axis=1)
    if 'A3_speed0' in res:
        res['zero_intensity_separator'] = res['A3_speed0'] >= zero_auc
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--features', default=None)
    ap.add_argument('--out', default='results/audit')
    args = ap.parse_args()

    feats_csv = args.features
    if feats_csv is None:
        from mspoof.config import load_config
        feats_csv = load_config(args.config).paths.features_csv
    df = pd.read_csv(feats_csv)
    res = audit(df)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, 'single_feature_auc_screen.csv'))

    print('=== ORACLE SUSPECTS (AUC >= 0.99 on a full scenario) ===')
    print(res[res['oracle_any_scenario']].round(4).to_string())
    if 'zero_intensity_separator' in res:
        print('\n=== ZERO-INTENSITY SEPARATORS (A3 speed 0, AUC >= 0.75) ===')
        z = res[res['zero_intensity_separator']].sort_values('A3_speed0',
                                                             ascending=False)
        print(z.round(4).to_string())
    print('\n=== FAMILY SUMMARY (max AUC) ===')
    num_cols = [c for c in res.columns
                if c not in ('family', 'oracle_any_scenario',
                             'zero_intensity_separator')]
    print(res.groupby('family')[num_cols].max().round(4).to_string())
    print(f'\nSaved -> {args.out}/single_feature_auc_screen.csv')


if __name__ == '__main__':
    main()
