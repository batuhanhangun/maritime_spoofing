#!/usr/bin/env python
"""Generalization sweep + threshold study (the journal paper's core MARSIM
experiments). Resume-capable: each unit of work writes one part-CSV under
<results>/sweep/parts/ and is skipped if it already exists, so interrupted
runs continue where they left off.

Stages (run all on a normal PC; full sweep is roughly 1-2 h with RF):
  presets   repetition split + all param_holdout presets, multi-seed,
            RF/XGB/LGBM (+SVM if enabled) and MANA on identical test rows
  loso      leave-one-speed-out over A3 shift_speed (interpolation regime)
  extrap    per-speed recall inside the a3_speed_low EXTRAPOLATION regime
  thresh    threshold-policy study in the extrapolation regime
            (default 0.5 / val_max_f1 / fpr_target / test-oracle ceiling)
  report    merge parts, write summary tables + the two paper figures

Usage:
    python scripts/run_generalization_sweep.py --config config.yaml --stage all
    python scripts/run_generalization_sweep.py --stage loso --models gbm
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mspoof.config import load_config          # noqa: E402
from mspoof import splits as sp                # noqa: E402
from mspoof.ablation import drop_families      # noqa: E402
from mspoof.preprocess import iterative_corr_filter  # noqa: E402
from mspoof.thresholds import pick_threshold, scores_of  # noqa: E402

META = ['filename', 'scenario', 'label', 'index',
        'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value',
        'mana_pred', '_error']
PRESETS = ['a3_speed_low', 'a3_speed_high', 'a3_angle_band', 'a2_delay_low']


# ---------------------------------------------------------------------------

def load_data(cfg, ablate):
    df = pd.read_csv(cfg.paths.features_csv)
    if '_error' in df.columns:
        df = df[df['_error'].fillna('') == ''].reset_index(drop=True)
    if os.path.exists(cfg.paths.mana_predictions_csv):
        mana = pd.read_csv(cfg.paths.mana_predictions_csv)[
            ['filename', 'mana_pred']]
        df = df.merge(mana, on='filename', how='left')
    else:
        df['mana_pred'] = np.nan
    cols = [c for c in df.columns if c not in META]
    cols = drop_families(cols, ablate)
    return df, cols


def prep(Xtr, Xte, corr_thr=0.95):
    std = Xtr.std()
    drop = std[std < 1e-10].index
    Xtr, Xte = Xtr.drop(columns=drop), Xte.drop(columns=drop)
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    drop = iterative_corr_filter(Xtr, corr_thr)
    return Xtr.drop(columns=drop), Xte.drop(columns=drop)


def model_zoo(which, seed):
    zoo = []
    if which in ('rf', 'all'):
        zoo.append(('RF', RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1)))
    if which in ('gbm', 'all'):
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        zoo.append(('XGB', XGBClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1,
            eval_metric='logloss')))
        zoo.append(('LGBM', LGBMClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1, verbose=-1)))
    return zoo


def get_split(df, tag):
    if tag == 'repetition':
        return sp.repetition_split(df)
    if tag.startswith('preset:'):
        return sp.param_holdout(df, **sp.PARAM_HOLDOUT_PRESETS[tag[7:]])
    if tag.startswith('speed:'):
        return sp.param_holdout(df, scenario='A3', param_name='shift_speed',
                                test_values=[float(tag[6:])])
    raise ValueError(tag)


def part_path(parts_dir, name):
    return os.path.join(parts_dir, f'{name}.csv')


def eval_unit(df, cols, tag, which, seeds, parts_dir, extra=None):
    """One (split, model-group) unit -> one part file. MANA rides along in
    the 'rf' group so it is computed exactly once per split."""
    out = part_path(parts_dir, f'{tag.replace(":", "_")}__{which}')
    if os.path.exists(out):
        print(f'  skip (exists): {os.path.basename(out)}')
        return
    tr, te = get_split(df, tag)
    y = (df['label'] == 'spoofed').astype(int).values
    yte = y[te]
    rows = []
    name = 'loso_speed' if tag.startswith('speed:') else tag
    if which != 'mana':
        Xtr, Xte = prep(df.loc[tr, cols].copy(), df.loc[te, cols].copy())
        ytr = y[tr]
        for seed in seeds:
            for mname, mdl in model_zoo(which, seed):
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                prob = scores_of(mdl, Xte)
                rows.append({
                    'split': name, 'model': mname, 'seed': seed,
                    'f1': f1_score(yte, pred),
                    'precision': precision_score(yte, pred, zero_division=0),
                    'recall': recall_score(yte, pred),
                    'auc': roc_auc_score(yte, prob) if len(set(yte)) > 1 else np.nan,
                    'fpr': float(((pred == 1) & (yte == 0)).sum()) / max((yte == 0).sum(), 1),
                    **(extra or {})})
    else:
        mp = df.loc[te, 'mana_pred'].values
        valid = pd.Series(mp).isin([0, 1]).values
        rows.append({
            'split': name, 'model': 'MANA', 'seed': -1,
            'f1': f1_score(yte[valid], mp[valid]),
            'precision': precision_score(yte[valid], mp[valid], zero_division=0),
            'recall': recall_score(yte[valid], mp[valid]),
            'auc': np.nan,
            'fpr': float(((mp[valid] == 1) & (yte[valid] == 0)).sum()) / max((yte[valid] == 0).sum(), 1),
            'n_excluded': int((~valid).sum()), **(extra or {})})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'  wrote {os.path.basename(out)}')


# ---------------------------------------------------------------------------

def stage_presets(df, cols, seeds, parts_dir, models):
    for tag in ['repetition'] + [f'preset:{p}' for p in PRESETS]:
        for which in models + ['mana']:
            eval_unit(df, cols, tag, which, seeds, parts_dir)


def stage_loso(df, cols, seeds, parts_dir, models):
    speeds = sorted(df.loc[(df.scenario == 'A3') &
                           (df.param_2_name == 'shift_speed'),
                           'param_2_value'].unique())
    for v in speeds:
        for which in models + ['mana']:
            eval_unit(df, cols, f'speed:{v}', which, seeds, parts_dir,
                      extra={'held_out_speed': v})


def stage_extrap(df, cols, seeds, parts_dir, models):
    """Per-speed recall INSIDE the extrapolation regime (train speed>16)."""
    out = part_path(parts_dir, 'extrapolation_per_speed')
    if os.path.exists(out):
        print('  skip (exists): extrapolation_per_speed.csv')
        return
    tr, te = sp.param_holdout(df, **sp.PARAM_HOLDOUT_PRESETS['a3_speed_low'])
    y = (df['label'] == 'spoofed').astype(int).values
    Xtr, Xte = prep(df.loc[tr, cols].copy(), df.loc[te, cols].copy())
    test = df.loc[te].copy()
    test['y'] = y[te]
    votes = {}
    for which in models:
        for mname, _ in model_zoo(which, 0):
            votes[mname] = np.zeros(len(test))
    n_seeds = {m: 0 for m in votes}
    for seed in seeds:
        for which in models:
            for mname, mdl in model_zoo(which, seed):
                mdl.fit(Xtr, y[tr])
                votes[mname] += mdl.predict(Xte)
                n_seeds[mname] += 1
    rows = []
    for v, g in test[test.y == 1].groupby('param_2_value'):
        row = {'speed': v}
        for mname in votes:
            frac = votes[mname][test.index.get_indexer(g.index)] / n_seeds[mname]
            row[mname] = float((frac >= 0.5).mean())
        row['MANA'] = float((g['mana_pred'] == 1).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('  wrote extrapolation_per_speed.csv')


def stage_thresh(df, cols, parts_dir, models, target_fprs=(0.01, 0.05),
                 seed=42, val_fraction=0.25):
    out = part_path(parts_dir, 'threshold_study')
    if os.path.exists(out):
        print('  skip (exists): threshold_study.csv')
        return
    tr, te = sp.param_holdout(df, **sp.PARAM_HOLDOUT_PRESETS['a3_speed_low'])
    y = (df['label'] == 'spoofed').astype(int).values
    X_all, Xte = prep(df.loc[tr, cols].copy(), df.loc[te, cols].copy())
    yte = y[te]
    Xfit, Xval, yfit, yval = train_test_split(
        X_all, y[tr], test_size=val_fraction, stratify=y[tr],
        random_state=seed)
    rows = []
    for which in models:
        for mname, mdl in model_zoo(which, seed):
            mdl.fit(Xfit, yfit)
            sval, ste = scores_of(mdl, Xval), scores_of(mdl, Xte)
            pols = {'default_0.5': pick_threshold(sval, yval, 'default'),
                    'val_max_f1': pick_threshold(sval, yval, 'val_max_f1')}
            for tf in target_fprs:
                pols[f'fpr_target_{tf:g}'] = pick_threshold(
                    sval, yval, 'fpr_target', target_fpr=tf)
            p, r, th = precision_recall_curve(yte, ste)
            f1o = 2 * p * r / np.clip(p + r, 1e-12, None)
            pols['oracle_test_ceiling'] = float(th[int(np.argmax(f1o[:-1]))])
            for pol, t in pols.items():
                pred = (ste >= t).astype(int)
                rows.append({'model': mname, 'policy': pol,
                             'threshold': float(t),
                             'recall': float(pred[yte == 1].mean()),
                             'fpr': float(pred[yte == 0].mean()),
                             'f1': f1_score(yte, pred),
                             'auc_test': roc_auc_score(yte, ste)})
    pd.DataFrame(rows).to_csv(out, index=False)
    print('  wrote threshold_study.csv')


def stage_report(parts_dir, out_dir):
    parts = [pd.read_csv(os.path.join(parts_dir, f))
             for f in sorted(os.listdir(parts_dir)) if f.endswith('.csv')
             and not f.startswith(('extrapolation', 'threshold'))]
    if parts:
        allr = pd.concat(parts, ignore_index=True)
        allr.to_csv(os.path.join(out_dir, 'sweep_all.csv'), index=False)
        pres = allr[allr.split != 'loso_speed']
        summ = pres.groupby(['split', 'model'])[
            ['f1', 'precision', 'recall', 'fpr']].agg(['mean', 'std']).round(4)
        summ.to_csv(os.path.join(out_dir, 'preset_summary.csv'))
        print(summ.to_string())
        loso = allr[allr.split == 'loso_speed']
        if len(loso):
            piv = loso.groupby(['model', 'held_out_speed'])['recall'].mean()\
                      .unstack(0).round(4)
            piv.to_csv(os.path.join(out_dir, 'loso_recall_by_speed.csv'))

    # figures (best effort; headless-safe)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = {'RF': '#1f77b4', 'XGB': '#d62728',
                  'LGBM': '#ff7f0e', 'MANA': '#2ca02c'}
        ext_p = os.path.join(parts_dir, 'extrapolation_per_speed.csv')
        if len(loso) and os.path.exists(ext_p):
            ext = pd.read_csv(ext_p)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
            for m in piv.columns:
                axes[0].plot(piv.index, piv[m], marker='o', ms=4,
                             color=colors.get(m), label=m)
            axes[0].set_title('(a) Interpolation: leave-one-speed-out')
            axes[0].set_xlabel('Held-out shift speed')
            axes[0].set_ylabel('Recall')
            axes[0].set_ylim(-0.03, 1.03)
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            for m in [c for c in ext.columns if c != 'speed']:
                axes[1].plot(ext.speed, ext[m], marker='s', ms=4,
                             color=colors.get(m), label=m)
            axes[1].set_title('(b) Extrapolation: train speed > 16')
            axes[1].set_xlabel('Test shift speed')
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, 'fig_generalization_cliffs.png'),
                        dpi=200, bbox_inches='tight')
            print('wrote fig_generalization_cliffs.png')
    except ImportError:
        print('matplotlib not installed; skipping figures')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--stage', default='all',
                    choices=['presets', 'loso', 'extrap', 'thresh',
                             'report', 'all'])
    ap.add_argument('--models', default='rf,gbm',
                    help='comma list among rf,gbm (MANA always included)')
    ap.add_argument('--seeds', default='42,7,123',
                    help='comma list of model seeds')
    args = ap.parse_args()

    cfg = load_config(args.config)
    ablate = list(getattr(cfg.preprocessing, 'ablate_families', []) or [])
    seeds = [int(s) for s in args.seeds.split(',')]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    out_dir = os.path.join(cfg.paths.results_dir, 'sweep')
    parts_dir = os.path.join(out_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)

    df, cols = load_data(cfg, ablate)
    print(f'features after ablation({ablate}): {len(cols)}')

    t0 = time.time()
    if args.stage in ('presets', 'all'):
        stage_presets(df, cols, seeds, parts_dir, models)
    if args.stage in ('loso', 'all'):
        stage_loso(df, cols, seeds, parts_dir, models)
    if args.stage in ('extrap', 'all'):
        stage_extrap(df, cols, seeds, parts_dir, models)
    if args.stage in ('thresh', 'all'):
        stage_thresh(df, cols, parts_dir, models)
    if args.stage in ('report', 'all'):
        stage_report(parts_dir, out_dir)
    print(f'done in {time.time() - t0:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
