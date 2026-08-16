#!/usr/bin/env python
"""Train and evaluate classifiers on the MARSIM feature table.

Changes vs the original train_classifiers.py:
* split regimes: repetition (paper), kfold (variance), param_holdout
  (generalization to unseen attack parameters; Review 1's key concern)
* model tiers: "default" (true library defaults, n_estimators=500) and
  "adjusted" (the original repo's hand-tuned settings), so the paper's
  out-of-the-box claim and the stronger config are both reported honestly
* SVM trains on the FULL training set by default (the old 10k subsample and
  C=10 contradicted the paper text; subsampling remains available via config
  and is recorded in the output when used)
* MANA rows with errors/missing predictions are EXCLUDED from the comparison
  (symmetrically for all classifiers) instead of silently coerced to 0,
  and the exclusion count is reported
* base-rate analysis: precision as a function of assumed spoofing prevalence,
  recomputed from confusion counts (Review 1, deployment realism)
* every run writes a run_manifest.json capturing config, split, tier, seed,
  and library versions for artifact-grade reproducibility

Figures intentionally live in a separate plotting script; this one produces
CSV outputs only, so HPC runs need no display stack.

Usage:
    python scripts/train_classifiers.py --config config.yaml
    python scripts/train_classifiers.py --config config.yaml --split param_holdout --preset a3_speed_low
    python scripts/train_classifiers.py --config config.yaml --split kfold
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as sk_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mspoof.config import load_config  # noqa: E402
from mspoof import splits as split_mod  # noqa: E402

META_COLS = ['filename', 'scenario', 'label', 'index',
             'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value']


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def build_models(tier, seed, svm_cfg):
    """Return {name: (estimator, needs_scaling)} for the requested tier."""
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    zoo = {}
    if tier in ('default', 'both'):
        zoo['RF-default'] = (RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1), False)
        zoo['XGB-default'] = (XGBClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1,
            eval_metric='logloss'), False)
        zoo['LGBM-default'] = (LGBMClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1, verbose=-1), False)
        if svm_cfg.enabled:
            zoo['SVM-default'] = (SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=seed), True)
    if tier in ('adjusted', 'both'):
        zoo['RF-adjusted'] = (RandomForestClassifier(
            n_estimators=500, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=seed, n_jobs=-1), False)
        zoo['XGB-adjusted'] = (XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, random_state=seed,
            n_jobs=-1, eval_metric='logloss'), False)
        zoo['LGBM-adjusted'] = (LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, random_state=seed,
            n_jobs=-1, verbose=-1), False)
        if svm_cfg.enabled:
            zoo['SVM-adjusted'] = (SVC(
                kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
                probability=True, random_state=seed), True)
    return zoo


# ---------------------------------------------------------------------------
# Preprocessing (fit on train only; every step logged)
# ---------------------------------------------------------------------------

def preprocess(X_train, X_test, pp_cfg, log):
    std = X_train.std()
    zero_var = std[std < pp_cfg.zero_variance_eps].index.tolist()
    X_train = X_train.drop(columns=zero_var)
    X_test = X_test.drop(columns=zero_var)
    log.append({'step': 'zero_variance', 'n_dropped': len(zero_var),
                'columns': ','.join(zero_var)})

    nan_rate = X_train.isna().mean()
    nan_cols = nan_rate[nan_rate > pp_cfg.nan_col_drop_threshold].index.tolist()
    X_train = X_train.drop(columns=nan_cols)
    X_test = X_test.drop(columns=nan_cols)
    log.append({'step': 'nan_columns', 'n_dropped': len(nan_cols),
                'columns': ','.join(nan_cols)})

    imputer = SimpleImputer(strategy='median').fit(X_train)
    X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

    sk = X_train.apply(skew)
    skewed = sk[sk.abs() > pp_cfg.skew_threshold].index.tolist()
    for col in skewed:
        for D in (X_train, X_test):
            D[col] = np.sign(D[col]) * np.log1p(D[col].abs())
    log.append({'step': 'log_transform', 'n_dropped': 0, 'columns': ','.join(skewed)})

    if pp_cfg.winsorize:
        lo_q, hi_q = pp_cfg.winsorize_quantiles
        lo, hi = X_train.quantile(lo_q), X_train.quantile(hi_q)
        X_train = X_train.clip(lower=lo, upper=hi, axis=1)
        X_test = X_test.clip(lower=lo, upper=hi, axis=1)
        log.append({'step': 'winsorize', 'n_dropped': 0, 'columns': f'{lo_q},{hi_q}'})

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_cols = [c for c in upper.columns if any(upper[c] > pp_cfg.correlation_threshold)]
    X_train = X_train.drop(columns=corr_cols)
    X_test = X_test.drop(columns=corr_cols)
    log.append({'step': 'correlation', 'n_dropped': len(corr_cols),
                'columns': ','.join(corr_cols)})

    scaler = StandardScaler().fit(X_train)
    X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    return X_train, X_test, X_train_sc, X_test_sc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_row(name, y_true, pred, prob=None, extra=None):
    row = {'classifier': name,
           'accuracy': accuracy_score(y_true, pred),
           'precision': precision_score(y_true, pred, zero_division=0),
           'recall': recall_score(y_true, pred, zero_division=0),
           'f1': f1_score(y_true, pred, zero_division=0)}
    row['auc_roc'] = roc_auc_score(y_true, prob) if prob is not None and len(set(y_true)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if extra:
        row.update(extra)
    return row


def base_rate_analysis(rows, prevalences=(0.5, 0.1, 0.01, 1e-3, 1e-4)):
    """Precision at assumed spoofing prevalence pi, from TPR and FPR.

    precision(pi) = TPR*pi / (TPR*pi + FPR*(1 - pi)). Answers 'what fraction
    of alarms are real if spoofing is rare', without retraining anything.
    """
    out = []
    for r in rows:
        tpr = r['tp'] / (r['tp'] + r['fn']) if (r['tp'] + r['fn']) else np.nan
        fpr = r['fp'] / (r['fp'] + r['tn']) if (r['fp'] + r['tn']) else np.nan
        for pi in prevalences:
            denom = tpr * pi + fpr * (1 - pi)
            out.append({'classifier': r['classifier'], 'prevalence': pi,
                        'tpr': tpr, 'fpr': fpr,
                        'precision_at_prevalence': (tpr * pi / denom) if denom else np.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# One split evaluation
# ---------------------------------------------------------------------------

def run_split(df, train_mask, test_mask, cfg, split_name, out_dir, mana_df=None):
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = [c for c in df.columns if c not in META_COLS + ['_error']]

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = (train_df['label'] == 'spoofed').astype(int).values
    y_test = (test_df['label'] == 'spoofed').astype(int).values

    pp_log = []
    X_train, X_test, X_train_sc, X_test_sc = preprocess(
        X_train, X_test, cfg.preprocessing, pp_log)
    pd.DataFrame(pp_log).to_csv(os.path.join(out_dir, 'preprocessing_log.csv'), index=False)
    with open(os.path.join(out_dir, 'final_features.txt'), 'w') as fh:
        fh.write('\n'.join(X_train.columns))

    print(f'[{split_name}] train={len(X_train)} test={len(X_test)} '
          f'features={X_train.shape[1]} '
          f'(spoofed test={int(y_test.sum())}/{len(y_test)})')

    zoo = build_models(cfg.models.tier, cfg.models.seed, cfg.models.svm)
    rows, per_scenario_rows, preds = [], [], {}

    for name, (model, needs_scaling) in zoo.items():
        Xtr = X_train_sc if needs_scaling else X_train
        Xte = X_test_sc if needs_scaling else X_test
        ytr = y_train
        subsampled = False
        if name.startswith('SVM') and not cfg.models.svm.train_on_full_set \
                and len(Xtr) > cfg.models.svm.subsample_size:
            Xtr, _, ytr, _ = sk_split(Xtr, ytr,
                                      train_size=cfg.models.svm.subsample_size,
                                      random_state=cfg.models.seed, stratify=ytr)
            subsampled = True

        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        t0 = time.time()
        pred = model.predict(Xte)
        infer_ms = (time.time() - t0) / max(len(Xte), 1) * 1000
        prob = model.predict_proba(Xte)[:, 1]
        preds[name] = pred

        rows.append(metric_row(name, y_test, pred, prob,
                               {'train_time_s': fit_s,
                                'inference_ms_per_sample': infer_ms,
                                'svm_subsampled': subsampled,
                                'n_train': len(Xtr)}))
        for scen in sorted(test_df['scenario'].unique()):
            m = (test_df['scenario'] == scen).values
            if m.sum():
                per_scenario_rows.append(metric_row(
                    name, y_test[m], pred[m], prob[m], {'scenario': scen}))
        print(f'  {name:<14} F1={rows[-1]["f1"]:.4f} P={rows[-1]["precision"]:.4f} '
              f'R={rows[-1]["recall"]:.4f} (fit {fit_s:.0f}s)')

    # MANA joins the comparison only on rows where it produced a prediction.
    if mana_df is not None:
        merged = test_df.merge(mana_df, on='filename', how='left')
        valid = merged['mana_pred'].isin([0, 1]).values
        n_excluded = int((~valid).sum())
        if n_excluded:
            print(f'  MANA: excluding {n_excluded} test rows with missing/error predictions')
        rows.append(metric_row('MANA', y_test[valid],
                               merged.loc[valid, 'mana_pred'].astype(int).values,
                               None, {'n_excluded': n_excluded}))
        for scen in sorted(test_df['scenario'].unique()):
            m = valid & (merged['scenario'] == scen).values
            if m.sum():
                per_scenario_rows.append(metric_row(
                    'MANA', y_test[m], merged.loc[m, 'mana_pred'].astype(int).values,
                    None, {'scenario': scen}))

    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(out_dir, 'overall_results.csv'), index=False)
    pd.DataFrame(per_scenario_rows).to_csv(
        os.path.join(out_dir, 'per_scenario_results.csv'), index=False)
    base_rate_analysis(rows).to_csv(
        os.path.join(out_dir, 'base_rate_analysis.csv'), index=False)

    # Per-parameter breakdown for cliff curves and heatmaps (test side only).
    pp = test_df[META_COLS].copy()
    pp['y_true'] = y_test
    for name, pred in preds.items():
        pp[f'pred_{name}'] = pred
    pp.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    # McNemar vs MANA where available
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        if mana_df is not None:
            merged = test_df.merge(mana_df, on='filename', how='left')
            valid = merged['mana_pred'].isin([0, 1]).values
            mrows = []
            for name, pred in preds.items():
                a = (pred[valid] == y_test[valid])
                b = (merged.loc[valid, 'mana_pred'].astype(int).values == y_test[valid])
                table = [[int((a & b).sum()), int((a & ~b).sum())],
                         [int((~a & b).sum()), int((~a & ~b).sum())]]
                res = mcnemar(table, exact=(table[0][1] + table[1][0]) < 25)
                mrows.append({'classifier': name, 'vs': 'MANA', 'p_value': res.pvalue})
            pd.DataFrame(mrows).to_csv(os.path.join(out_dir, 'mcnemar.csv'), index=False)
    except ImportError:
        pass

    return overall


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--split', default=None,
                    choices=['repetition', 'kfold', 'param_holdout'])
    ap.add_argument('--preset', default=None,
                    help='param_holdout preset name (see mspoof.splits)')
    ap.add_argument('--tier', default=None, choices=['default', 'adjusted', 'both'])
    ap.add_argument('--profile-tag', default='',
                    help='suffix for the results directory (e.g. deployable)')
    args = ap.parse_args()

    cfg = load_config(args.config, **{
        'split.regime': args.split,
        'split.param_holdout_preset': args.preset,
        'models.tier': args.tier,
    })

    df = pd.read_csv(cfg.paths.features_csv)
    if '_error' in df.columns:
        n_bad = int((df['_error'].fillna('') != '').sum())
        if n_bad:
            print(f'Dropping {n_bad} rows with extraction errors')
            df = df[df['_error'].fillna('') == ''].reset_index(drop=True)

    mana_df = None
    if os.path.exists(cfg.paths.mana_predictions_csv):
        mana_df = pd.read_csv(cfg.paths.mana_predictions_csv)
    else:
        print('NOTE: MANA predictions not found; ML-only run')

    regime = cfg.split.regime
    tag = f'_{args.profile_tag}' if args.profile_tag else ''
    root = os.path.join(cfg.paths.results_dir, f'{regime}{tag}')

    manifest = {'regime': regime, 'tier': cfg.models.tier, 'seed': cfg.models.seed,
                'features_csv': cfg.paths.features_csv,
                'python': platform.python_version()}

    if regime == 'repetition':
        tr, te = split_mod.repetition_split(df)
        run_split(df, tr, te, cfg, 'repetition', root, mana_df)
    elif regime == 'kfold':
        fold_frames = []
        for fold, tr, te in split_mod.kfold_repetition(df, cfg.split.kfold_n):
            res = run_split(df, tr, te, cfg, f'fold{fold}',
                            os.path.join(root, f'fold{fold}'), mana_df)
            res['fold'] = fold
            fold_frames.append(res)
        allf = pd.concat(fold_frames, ignore_index=True)
        summary = allf.groupby('classifier')[['f1', 'precision', 'recall']].agg(['mean', 'std'])
        summary.to_csv(os.path.join(root, 'kfold_summary.csv'))
        print('\nK-fold summary (mean +/- std):')
        print(summary.round(4).to_string())
    elif regime == 'param_holdout':
        preset_name = cfg.split.param_holdout_preset
        preset = split_mod.PARAM_HOLDOUT_PRESETS[preset_name]
        manifest['preset'] = {**preset, 'name': preset_name}
        tr, te = split_mod.param_holdout(df, **preset)
        run_split(df, tr, te, cfg, f'param_holdout:{preset_name}',
                  os.path.join(root, preset_name), mana_df)
    else:
        raise SystemExit(f'unknown split regime {regime!r}')

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'run_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'\nResults -> {root}')


if __name__ == '__main__':
    main()
