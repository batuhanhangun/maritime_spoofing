#!/usr/bin/env python
"""
train_classifiers.py — MARSIM ML Classification Pipeline.

Trains 4 ML classifiers (RF, XGBoost, LightGBM, SVM), loads pre-computed MANA
predictions, evaluates all 5 classifiers, and generates results tables + figures.

Usage:
    conda activate marsim_env
    python train_classifiers.py [--skip-svm]
"""

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description='MARSIM ML Classification Pipeline')
parser.add_argument('--skip-svm', action='store_true', help='Skip SVM training (slow)')
args = parser.parse_args()

# ============================================================
# Figure styling (publication quality)
# ============================================================
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ============================================================
# Paths
# ============================================================
INPUT_CSV = os.path.join("data", "marsim_features.csv")
MANA_CSV = os.path.join("data", "mana_predictions.csv")
RESULTS_DIR = os.path.join("results", "ml_classifiers")
os.makedirs(RESULTS_DIR, exist_ok=True)

METADATA_COLS = ['filename', 'scenario', 'label', 'index',
                 'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value']

# ============================================================
# 1. Load CSV
# ============================================================
print("=" * 70)
print("MARSIM ML Classification Pipeline")
print("=" * 70)
print(f"\n[1/10] Loading data from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"  Shape: {df.shape}")
print(f"  Scenarios: {df['scenario'].value_counts().to_dict()}")
print(f"  Labels: {df['label'].value_counts().to_dict()}")

# ============================================================
# 2. Train/Test Split by index (BEFORE preprocessing)
# ============================================================
print(f"\n[2/10] Train/Test split by index column (train: 0-15, test: 16-19)")
df['index'] = df['index'].astype(int)
train_df = df[df['index'].isin(range(0, 16))].reset_index(drop=True)
test_df = df[df['index'].isin(range(16, 20))].reset_index(drop=True)
print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

# Separate features and metadata
feature_cols = [c for c in df.columns if c not in METADATA_COLS]
print(f"  Feature columns: {len(feature_cols)}")

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = (train_df['label'] == 'spoofed').astype(int)
y_test = (test_df['label'] == 'spoofed').astype(int)
train_meta = train_df[METADATA_COLS].reset_index(drop=True)
test_meta = test_df[METADATA_COLS].reset_index(drop=True)

print(f"  Train label balance: spoofed={y_train.sum()}, unspoofed={(y_train==0).sum()}")
print(f"  Test label balance:  spoofed={y_test.sum()}, unspoofed={(y_test==0).sum()}")

# ============================================================
# 3. Preprocessing Pipeline
# ============================================================
preprocessing_log = []

# Step 3a: Drop zero-variance features
print(f"\n[3/10] Preprocessing pipeline")
print(f"  Step 3a: Dropping zero/near-zero variance features...")
train_std = X_train.std()
zero_var_cols = train_std[train_std < 1e-10].index.tolist()
print(f"    Dropped {len(zero_var_cols)} columns: {zero_var_cols}")
preprocessing_log.append({'step': 'zero_variance_removal', 'columns_dropped': ', '.join(zero_var_cols),
                          'features_remaining': len(feature_cols) - len(zero_var_cols)})
X_train = X_train.drop(columns=zero_var_cols)
X_test = X_test.drop(columns=zero_var_cols)
n_after_zv = X_train.shape[1]
print(f"    Features remaining: {n_after_zv}")

# Step 3b: Handle NaN values
print(f"  Step 3b: Handling NaN values...")
nan_rate_before = X_train.isna().mean()
nan_total_before = X_train.isna().sum().sum()
print(f"    Total NaN values in train: {nan_total_before}")

nan_dropped_cols = nan_rate_before[nan_rate_before > 0.5].index.tolist()
if nan_dropped_cols:
    print(f"    Dropping {len(nan_dropped_cols)} columns with >50% NaN: {nan_dropped_cols}")
    X_train = X_train.drop(columns=nan_dropped_cols)
    X_test = X_test.drop(columns=nan_dropped_cols)
preprocessing_log.append({'step': 'nan_column_removal', 'columns_dropped': ', '.join(nan_dropped_cols),
                          'features_remaining': X_train.shape[1]})

imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)
X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
nan_total_after = X_train.isna().sum().sum()
print(f"    NaN after imputation: {nan_total_after}")
n_after_nan = X_train.shape[1]

# Step 3c: Log-transform heavily skewed features
print(f"  Step 3c: Log-transforming heavily skewed features...")
skewness = X_train.apply(skew)
highly_skewed = skewness[skewness.abs() > 5.0].index.tolist()
print(f"    Highly skewed features (|skew| > 5): {len(highly_skewed)}")
log_transformed = []
for col in highly_skewed:
    if X_train[col].min() >= 0:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
        log_transformed.append(col)
    else:
        X_train[col] = np.sign(X_train[col]) * np.log1p(X_train[col].abs())
        X_test[col] = np.sign(X_test[col]) * np.log1p(X_test[col].abs())
        log_transformed.append(col)
preprocessing_log.append({'step': 'log_transform', 'columns_dropped': ', '.join(log_transformed),
                          'features_remaining': X_train.shape[1]})
skewness_after = X_train[log_transformed].apply(skew) if log_transformed else pd.Series()
print(f"    Transformed {len(log_transformed)} features")

# Step 3d: Outlier capping (Winsorization)
print(f"  Step 3d: Winsorizing outliers (1st/99th percentile)...")
lower = X_train.quantile(0.01)
upper = X_train.quantile(0.99)
X_train = X_train.clip(lower=lower, upper=upper, axis=1)
X_test = X_test.clip(lower=lower, upper=upper, axis=1)

# Step 3e: Remove highly correlated features
print(f"  Step 3e: Removing highly correlated features (r > 0.95)...")
corr_matrix = X_train.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corr_dropped = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
print(f"    Dropping {len(corr_dropped)} correlated features")
preprocessing_log.append({'step': 'correlation_removal', 'columns_dropped': ', '.join(corr_dropped),
                          'features_remaining': X_train.shape[1] - len(corr_dropped)})
X_train = X_train.drop(columns=corr_dropped)
X_test = X_test.drop(columns=corr_dropped)
n_final = X_train.shape[1]
print(f"    Final feature count: {n_final}")

# Step 3f: Feature scaling (for SVM)
print(f"  Step 3f: Standard scaling (for SVM)...")
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

# Step 3g: Preprocessing summary
print(f"\n  {'='*50}")
print(f"  === PREPROCESSING SUMMARY ===")
print(f"  Original features:           {len(feature_cols)}")
print(f"  After zero-variance removal: {n_after_zv}")
print(f"  After NaN column removal:    {n_after_nan}")
print(f"  Log-transformed features:    {len(log_transformed)}")
print(f"  After correlation removal:   {n_final}")
print(f"  Final feature count:         {n_final}")
print(f"  Train / Test sizes:          {len(X_train)} / {len(X_test)}")
print(f"  Label balance (train):       spoofed={y_train.sum()}, unspoofed={(y_train==0).sum()}")
print(f"  Label balance (test):        spoofed={y_test.sum()}, unspoofed={(y_test==0).sum()}")
print(f"  {'='*50}")

# Step 3h: Save preprocessed data
print(f"\n  Saving preprocessed data...")
X_train.to_csv(os.path.join(RESULTS_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(RESULTS_DIR, 'X_test.csv'), index=False)
with open(os.path.join(RESULTS_DIR, 'final_features.txt'), 'w') as f:
    for col in X_train.columns:
        f.write(col + '\n')
pd.DataFrame(preprocessing_log).to_csv(os.path.join(RESULTS_DIR, 'preprocessing_summary.csv'), index=False)

# ============================================================
# 4. Load MANA predictions
# ============================================================
print(f"\n[4/10] Loading MANA predictions from {MANA_CSV}")
mana_df = pd.read_csv(MANA_CSV)
test_with_mana = test_meta.merge(mana_df, on='filename', how='left')
mana_test_pred = test_with_mana['mana_pred'].values

n_errors = (test_with_mana['mana_pred'] == -1).sum()
n_missing = test_with_mana['mana_pred'].isna().sum()
if n_errors > 0:
    print(f"  WARNING: {n_errors} MANA predictions are errors (-1), treating as unspoofed (0)")
    mana_test_pred = np.where(mana_test_pred == -1, 0, mana_test_pred).astype(int)
if n_missing > 0:
    print(f"  WARNING: {n_missing} files missing from mana_predictions.csv, treating as unspoofed (0)")
    mana_test_pred = np.nan_to_num(mana_test_pred, nan=0).astype(int)
print(f"  MANA predictions loaded: {len(mana_test_pred)}")

# ============================================================
# 5. Train ML classifiers
# ============================================================
print(f"\n[5/10] Training classifiers...")
classifiers = {}
train_times = {}

# Random Forest
print(f"  Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5,
                            min_samples_leaf=2, max_features='sqrt',
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
train_times['RF'] = time.time() - t0
classifiers['RF'] = rf
print(f"    Done in {train_times['RF']:.1f}s")

# XGBoost
print(f"  Training XGBoost...")
t0 = time.time()
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=1,
                     random_state=42, n_jobs=-1, eval_metric='logloss')
xgb.fit(X_train, y_train)
train_times['XGB'] = time.time() - t0
classifiers['XGB'] = xgb
print(f"    Done in {train_times['XGB']:.1f}s")

# LightGBM
print(f"  Training LightGBM...")
t0 = time.time()
lgbm = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, min_child_samples=20, random_state=42,
                       n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train)
train_times['LGBM'] = time.time() - t0
classifiers['LGBM'] = lgbm
print(f"    Done in {train_times['LGBM']:.1f}s")

# SVM
if not args.skip_svm:
    print(f"  Training SVM (subsampled to 10K)...")
    t0 = time.time()
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
              probability=True, random_state=42)
    X_train_svm, _, y_train_svm, _ = sklearn_train_test_split(
        X_train_scaled, y_train, train_size=10000, random_state=42, stratify=y_train)
    svm.fit(X_train_svm, y_train_svm)
    train_times['SVM'] = time.time() - t0
    classifiers['SVM'] = svm
    print(f"    Done in {train_times['SVM']:.1f}s")
else:
    print(f"  Skipping SVM (--skip-svm flag)")

# ============================================================
# 6. Evaluate all classifiers
# ============================================================
print(f"\n[6/10] Evaluating classifiers on test set...")

# Get predictions
predictions = {}
probabilities = {}
inference_times = {}

for name, clf in classifiers.items():
    X_eval = X_test_scaled if name == 'SVM' else X_test
    t0 = time.time()
    preds = clf.predict(X_eval)
    t_inf = time.time() - t0
    predictions[name] = preds
    inference_times[name] = (t_inf / len(X_eval)) * 1000  # ms per sample
    probabilities[name] = clf.predict_proba(X_eval)[:, 1]

predictions['MANA'] = mana_test_pred
inference_times['MANA'] = 249.0  # ms/file measured from run_mana_baseline.py (10787s / 43320 files)

all_clf_names = list(classifiers.keys()) + ['MANA']

# Overall metrics
print(f"\n  {'Classifier':<10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'ms/sample':>10}")
print(f"  {'-'*58}")
overall_rows = []
for name in all_clf_names:
    pred = predictions[name]
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    if name in probabilities:
        auc = roc_auc_score(y_test, probabilities[name])
    else:
        auc = np.nan
    inf_t = inference_times[name]
    overall_rows.append({'classifier': name, 'accuracy': acc, 'precision': prec,
                         'recall': rec, 'f1': f1, 'auc_roc': auc, 'inference_time_ms': inf_t})
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    inf_str = f"{inf_t:.4f}" if not np.isnan(inf_t) else "N/A"
    print(f"  {name:<10} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc_str:>7} {inf_str:>10}")

overall_df = pd.DataFrame(overall_rows)
overall_df.to_csv(os.path.join(RESULTS_DIR, 'overall_results.csv'), index=False)

# Confusion matrices
cm_rows = []
for name in all_clf_names:
    tn, fp, fn, tp = confusion_matrix(y_test, predictions[name]).ravel()
    cm_rows.append({'classifier': name, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
pd.DataFrame(cm_rows).to_csv(os.path.join(RESULTS_DIR, 'confusion_matrices.csv'), index=False)

# Per-scenario metrics
print(f"\n  Per-scenario results:")
scenario_rows = []
for name in all_clf_names:
    for scenario in ['A1', 'A2', 'A3']:
        mask = test_meta['scenario'] == scenario
        y_s = y_test[mask]
        p_s = predictions[name][mask]
        prec = precision_score(y_s, p_s, zero_division=0)
        rec = recall_score(y_s, p_s, zero_division=0)
        f1_val = f1_score(y_s, p_s, zero_division=0)
        if name in probabilities:
            auc_val = roc_auc_score(y_s, probabilities[name][mask])
        else:
            auc_val = np.nan
        scenario_rows.append({'classifier': name, 'scenario': scenario,
                              'precision': prec, 'recall': rec, 'f1': f1_val, 'auc_roc': auc_val})
        auc_s = f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A"
        print(f"    {name:<10} {scenario} — P={prec:.4f} R={rec:.4f} F1={f1_val:.4f} AUC={auc_s}")

pd.DataFrame(scenario_rows).to_csv(os.path.join(RESULTS_DIR, 'per_scenario_results.csv'), index=False)

# MANA per-method results
print(f"\n  MANA per-method breakdown:")
mana_method_rows = []
method_cols_map = {'pdm': 'MultipleReceiversMethod', 'pcc_sog': 'PhysicalSpeedLimitMethod',
                   'pcc_rot': 'PhysicalRateOfTurnLimitMethod', 'pcc_height': 'PhysicalHeightLimitMethod',
                   'cdm': 'TimeDriftMethod', 'cnm': 'CarrierToNoiseDensityMethod'}
for scenario in ['A1', 'A2', 'A3']:
    mask = test_with_mana['scenario'] == scenario
    sub = test_with_mana[mask]
    for short, full in method_cols_map.items():
        if short in sub.columns:
            spoofed_mask = sub['label'] == 'spoofed'
            triggered_on_spoofed = sub.loc[spoofed_mask, short].sum()
            triggered_on_unspoofed = sub.loc[~spoofed_mask, short].sum()
            mana_method_rows.append({'scenario': scenario, 'method': short, 'method_full': full,
                                     'triggered_spoofed': int(triggered_on_spoofed),
                                     'total_spoofed': int(spoofed_mask.sum()),
                                     'triggered_unspoofed': int(triggered_on_unspoofed),
                                     'total_unspoofed': int((~spoofed_mask).sum())})
            print(f"    {scenario} {short:>12}: spoofed={int(triggered_on_spoofed)}/{int(spoofed_mask.sum())}, "
                  f"unspoofed={int(triggered_on_unspoofed)}/{int((~spoofed_mask).sum())}")
pd.DataFrame(mana_method_rows).to_csv(os.path.join(RESULTS_DIR, 'mana_per_method_results.csv'), index=False)

# ============================================================
# 7. Feature importances
# ============================================================
print(f"\n[7/10] Computing feature importances...")
fi_rows = []
tree_models = {k: v for k, v in classifiers.items() if k in ['RF', 'XGB', 'LGBM']}
for name, clf in tree_models.items():
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:30]
    for rank, idx in enumerate(sorted_idx):
        fi_rows.append({'classifier': name, 'rank': rank + 1,
                        'feature': X_train.columns[idx], 'importance': importances[idx]})
pd.DataFrame(fi_rows).to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)

# ============================================================
# ROC curve data
# ============================================================
roc_rows = []
for name in classifiers.keys():
    if name in probabilities:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        for f_val, t_val in zip(fpr, tpr):
            roc_rows.append({'classifier': name, 'fpr': f_val, 'tpr': t_val})
# MANA single point
mana_tn, mana_fp, mana_fn, mana_tp = confusion_matrix(y_test, mana_test_pred).ravel()
mana_fpr = mana_fp / (mana_fp + mana_tn) if (mana_fp + mana_tn) > 0 else 0
mana_tpr = mana_tp / (mana_tp + mana_fn) if (mana_tp + mana_fn) > 0 else 0
roc_rows.append({'classifier': 'MANA', 'fpr': mana_fpr, 'tpr': mana_tpr})
pd.DataFrame(roc_rows).to_csv(os.path.join(RESULTS_DIR, 'roc_curve_data.csv'), index=False)

# ============================================================
# Test predictions
# ============================================================
test_pred_df = test_meta.copy()
test_pred_df['label_true_binary'] = y_test.values
for name in classifiers.keys():
    test_pred_df[f'{name}_pred'] = predictions[name]
    test_pred_df[f'{name}_prob'] = probabilities.get(name, np.nan)
test_pred_df['mana_pred'] = mana_test_pred
test_pred_df['mana_prob'] = np.nan
test_pred_df.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)

# ============================================================
# 8. Per-parameter heatmaps and curve data
# ============================================================
print(f"\n[8/10] Computing per-parameter heatmaps...")

def compute_heatmap_data(test_meta_df, y_true, preds_dict, scenario, p1_name, p2_name):
    """Compute F1 heatmap for a given scenario across two parameter axes."""
    mask = test_meta_df['scenario'] == scenario
    sub_meta = test_meta_df[mask].reset_index(drop=True)
    y_s = y_true[mask].values

    # Identify param columns
    p1_vals = sorted(sub_meta.loc[sub_meta['param_1_name'] == p1_name, 'param_1_value'].unique())
    p2_vals = sorted(sub_meta.loc[sub_meta['param_2_name'] == p2_name, 'param_2_value'].unique())

    # Fallback: check if params are swapped
    if len(p1_vals) == 0:
        p1_vals = sorted(sub_meta.loc[sub_meta['param_2_name'] == p1_name, 'param_2_value'].unique())
        p2_vals = sorted(sub_meta.loc[sub_meta['param_1_name'] == p2_name, 'param_1_value'].unique())

    all_rows = []
    clf_heatmaps = {name: np.full((len(p1_vals), len(p2_vals)), np.nan) for name in preds_dict}

    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            cell_mask = (
                ((sub_meta['param_1_name'] == p1_name) & (sub_meta['param_1_value'] == v1) &
                 (sub_meta['param_2_name'] == p2_name) & (sub_meta['param_2_value'] == v2)) |
                ((sub_meta['param_2_name'] == p1_name) & (sub_meta['param_2_value'] == v1) &
                 (sub_meta['param_1_name'] == p2_name) & (sub_meta['param_1_value'] == v2))
            )
            if cell_mask.sum() == 0:
                continue
            y_cell = y_s[cell_mask]
            row = {p1_name: v1, p2_name: v2}
            for name, pred_arr in preds_dict.items():
                p_cell = pred_arr[mask][cell_mask]
                if len(np.unique(y_cell)) > 1:
                    f1_val = f1_score(y_cell, p_cell, zero_division=0)
                else:
                    f1_val = float(np.mean(y_cell == p_cell))
                row[f'{name}_f1'] = f1_val
                clf_heatmaps[name][i, j] = f1_val
            all_rows.append(row)

    return pd.DataFrame(all_rows), clf_heatmaps, p1_vals, p2_vals

# Build combined predictions dict
all_preds = {name: predictions[name] for name in all_clf_names}

# A3: shift_angle × shift_speed
a3_data, a3_heatmaps, a3_p1, a3_p2 = compute_heatmap_data(
    test_meta, y_test, all_preds, 'A3', 'shift_angle', 'shift_speed')
a3_data.to_csv(os.path.join(RESULTS_DIR, 'a3_heatmap_data.csv'), index=False)

# A1: distance_to_ship × time_difference
a1_data, a1_heatmaps, a1_p1, a1_p2 = compute_heatmap_data(
    test_meta, y_test, all_preds, 'A1', 'distance_to_ship', 'time_difference')
a1_data.to_csv(os.path.join(RESULTS_DIR, 'a1_heatmap_data.csv'), index=False)

# A2: distance_to_ship × delay
a2_data, a2_heatmaps, a2_p1, a2_p2 = compute_heatmap_data(
    test_meta, y_test, all_preds, 'A2', 'distance_to_ship', 'delay')
a2_data.to_csv(os.path.join(RESULTS_DIR, 'a2_heatmap_data.csv'), index=False)

# Curve data: F1 and recall averaged over one parameter axis
def compute_curve_data(heatmap_df, param_col, clf_names):
    """Average F1 and recall across rows for each value of param_col."""
    rows = []
    for val in sorted(heatmap_df[param_col].unique()):
        sub = heatmap_df[heatmap_df[param_col] == val]
        row = {param_col: val}
        for name in clf_names:
            f1_col = f'{name}_f1'
            if f1_col in sub.columns:
                row[f'{name.lower()}_f1'] = sub[f1_col].mean()
        rows.append(row)
    return pd.DataFrame(rows)

a3_curve = compute_curve_data(a3_data, 'shift_speed', all_clf_names)
a3_curve.to_csv(os.path.join(RESULTS_DIR, 'a3_shift_speed_curve_data.csv'), index=False)

a1_curve = compute_curve_data(a1_data, 'time_difference', all_clf_names)
a1_curve.to_csv(os.path.join(RESULTS_DIR, 'a1_time_difference_curve_data.csv'), index=False)

a2_curve = compute_curve_data(a2_data, 'delay', all_clf_names)
a2_curve.to_csv(os.path.join(RESULTS_DIR, 'a2_delay_curve_data.csv'), index=False)

# Also compute recall-based curves for A3
def compute_recall_curve(test_meta_df, y_true, preds_dict, scenario, avg_param, val_param):
    """Compute recall per val_param value, averaged over avg_param."""
    mask = test_meta_df['scenario'] == scenario
    sub = test_meta_df[mask].reset_index(drop=True)
    y_s = y_true[mask].values
    vals = sorted(sub.loc[(sub['param_1_name'] == val_param), 'param_1_value'].unique())
    if len(vals) == 0:
        vals = sorted(sub.loc[(sub['param_2_name'] == val_param), 'param_2_value'].unique())
    rows = []
    for v in vals:
        cell = ((sub['param_1_name'] == val_param) & (sub['param_1_value'] == v)) | \
               ((sub['param_2_name'] == val_param) & (sub['param_2_value'] == v))
        y_cell = y_s[cell]
        # Only compute recall on spoofed samples
        spoofed_mask_cell = y_cell == 1
        row = {val_param: v}
        for name, pred_arr in preds_dict.items():
            p_cell = pred_arr[mask][cell]
            if spoofed_mask_cell.sum() > 0:
                rec = recall_score(y_cell, p_cell, zero_division=0)
            else:
                rec = np.nan
            row[f'{name.lower()}_recall'] = rec
        rows.append(row)
    return pd.DataFrame(rows)

a3_recall_curve = compute_recall_curve(test_meta, y_test, all_preds, 'A3', 'shift_angle', 'shift_speed')
# Merge recall columns into a3_curve
if not a3_recall_curve.empty:
    a3_curve_full = a3_curve.merge(a3_recall_curve, on='shift_speed', how='outer')
    a3_curve_full.to_csv(os.path.join(RESULTS_DIR, 'a3_shift_speed_curve_data.csv'), index=False)

# ============================================================
# 9. Generate all figures
# ============================================================
print(f"\n[9/10] Generating figures...")

CLF_COLORS = {'RF': '#2196F3', 'XGB': '#4CAF50', 'LGBM': '#FF9800', 'SVM': '#9C27B0', 'MANA': '#F44336'}
CLF_MARKERS = {'RF': 'o', 'XGB': 's', 'LGBM': '^', 'SVM': 'D', 'MANA': 'X'}

# Fig 1: Overall comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(all_clf_names))
width = 0.35
f1_vals = [overall_df.loc[overall_df['classifier'] == n, 'f1'].values[0] for n in all_clf_names]
auc_vals = [overall_df.loc[overall_df['classifier'] == n, 'auc_roc'].values[0] for n in all_clf_names]
bars1 = ax.bar(x - width/2, f1_vals, width, label='F1-Score', color='#2196F3', edgecolor='white')
# AUC bars only for ML classifiers
auc_bars = [v if not np.isnan(v) else 0 for v in auc_vals]
auc_colors = ['#4CAF50' if not np.isnan(v) else '#cccccc' for v in auc_vals]
bars2 = ax.bar(x + width/2, auc_bars, width, label='AUC-ROC', color=auc_colors, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(all_clf_names)
ax.set_ylabel('Score')
ax.set_title('Overall Classifier Comparison')
ax.set_ylim(0, 1.05)
ax.legend()
for bar, val in zip(bars1, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'overall_comparison_bar.png'))
plt.close()

# Fig 2: ROC curves
fig, ax = plt.subplots(figsize=(7, 7))
for name in classifiers.keys():
    if name in probabilities:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        auc_val = roc_auc_score(y_test, probabilities[name])
        ax.plot(fpr, tpr, color=CLF_COLORS[name], label=f'{name} (AUC={auc_val:.4f})', linewidth=2)
ax.plot(mana_fpr, mana_tpr, marker='X', markersize=14, color=CLF_COLORS['MANA'],
        label=f'MANA (FPR={mana_fpr:.3f}, TPR={mana_tpr:.3f})', linestyle='None', zorder=5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'))
plt.close()

# Fig 3: Individual A3 heatmaps
def plot_heatmap(data_matrix, p1_vals, p2_vals, title, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[f"{v}" for v in p2_vals],
                yticklabels=[f"{v}" for v in p1_vals], ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

for name in ['RF', 'XGB', 'LGBM', 'MANA']:
    if name in a3_heatmaps:
        plot_heatmap(a3_heatmaps[name], a3_p1, a3_p2,
                     f'A3 F1-Score — {name}', 'shift_speed', 'shift_angle',
                     os.path.join(RESULTS_DIR, f'a3_heatmap_{name.lower()}.png'))

# Fig 4: A3 heatmap comparison (2×2)
def plot_heatmap_comparison(heatmaps, p1_vals, p2_vals, scenario, xlabel, ylabel, ml_names, save_path):
    # Pick best ML model by overall F1
    best_ml = max(ml_names, key=lambda n: overall_df.loc[overall_df['classifier'] == n, 'f1'].values[0])
    models_to_show = [best_ml, 'MANA']
    if len(models_to_show) < 4:
        for n in ml_names:
            if n not in models_to_show:
                models_to_show.insert(1, n)
            if len(models_to_show) >= 4:
                break
    models_to_show = models_to_show[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, name in enumerate(models_to_show):
        ax = axes[idx // 2][idx % 2]
        if name in heatmaps:
            sns.heatmap(heatmaps[name], annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                        xticklabels=[f"{v}" for v in p2_vals],
                        yticklabels=[f"{v}" for v in p1_vals], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{scenario} F1-Score — {name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

ml_clf_names = [n for n in all_clf_names if n != 'MANA']
plot_heatmap_comparison(a3_heatmaps, a3_p1, a3_p2, 'A3', 'shift_speed', 'shift_angle',
                        ml_clf_names, os.path.join(RESULTS_DIR, 'a3_heatmap_comparison.png'))
plot_heatmap_comparison(a1_heatmaps, a1_p1, a1_p2, 'A1', 'time_difference', 'distance_to_ship',
                        ml_clf_names, os.path.join(RESULTS_DIR, 'a1_heatmap_comparison.png'))
plot_heatmap_comparison(a2_heatmaps, a2_p1, a2_p2, 'A2', 'delay', 'distance_to_ship',
                        ml_clf_names, os.path.join(RESULTS_DIR, 'a2_heatmap_comparison.png'))

# Fig 5: Feature importance (top 20 from best model)
best_tree = max(tree_models.keys(), key=lambda n: overall_df.loc[overall_df['classifier'] == n, 'f1'].values[0])
importances = tree_models[best_tree].feature_importances_
sorted_idx = np.argsort(importances)[::-1][:20]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(20), importances[sorted_idx][::-1], color='#2196F3')
ax.set_yticks(range(20))
ax.set_yticklabels([X_train.columns[i] for i in sorted_idx][::-1])
ax.set_xlabel('Feature Importance')
ax.set_title(f'Top 20 Features ({best_tree})')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_top20.png'))
plt.close()

# Fig 6: A3 shift_speed curve (KEY FIGURE)
fig, ax = plt.subplots(figsize=(10, 6))
for name in all_clf_names:
    col = f'{name.lower()}_f1'
    if col in a3_curve.columns:
        ax.plot(a3_curve['shift_speed'], a3_curve[col], marker=CLF_MARKERS.get(name, 'o'),
                color=CLF_COLORS.get(name, '#333'), label=name, linewidth=2, markersize=6)
ax.set_xlabel('Shift Speed')
ax.set_ylabel('F1-Score')
ax.set_title('A3 Detection Performance vs. Shift Speed (Averaged over Shift Angle)')
ax.legend()
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'a3_shift_speed_curve.png'))
plt.close()

# Fig 7: Confusion matrices grid
n_clfs = len(all_clf_names)
fig, axes = plt.subplots(1, n_clfs, figsize=(4 * n_clfs, 4))
if n_clfs == 1:
    axes = [axes]
for idx, name in enumerate(all_clf_names):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Unspoofed', 'Spoofed'], yticklabels=['Unspoofed', 'Spoofed'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
    axes[idx].set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'))
plt.close()

print(f"  All figures saved to {RESULTS_DIR}")

# ============================================================
# 10. McNemar's test (statistical significance)
# ============================================================
print(f"\n[10/10] Statistical significance tests (McNemar)...")
try:
    from statsmodels.stats.contingency_tables import mcnemar

    def mcnemar_test(y_true, pred_a, pred_b):
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)
        n_00 = int((correct_a & correct_b).sum())
        n_01 = int((correct_a & ~correct_b).sum())
        n_10 = int((~correct_a & correct_b).sum())
        n_11 = int((~correct_a & ~correct_b).sum())
        table = [[n_00, n_01], [n_10, n_11]]
        use_exact = (n_01 + n_10) < 25
        result = mcnemar(table, exact=use_exact)
        return result.pvalue

    mcnemar_rows = []
    # Overall: each ML classifier vs MANA
    for name in classifiers.keys():
        pval = mcnemar_test(y_test.values, predictions[name], mana_test_pred)
        mcnemar_rows.append({'classifier_a': name, 'classifier_b': 'MANA',
                             'subset': 'overall', 'p_value': pval,
                             'significant_at_005': pval < 0.05})
        print(f"  {name} vs MANA (overall): p={pval:.6f} {'***' if pval < 0.05 else ''}")

    # Per-scenario: ML vs MANA on A3
    for name in classifiers.keys():
        mask = test_meta['scenario'] == 'A3'
        pval = mcnemar_test(y_test[mask].values, predictions[name][mask], mana_test_pred[mask])
        mcnemar_rows.append({'classifier_a': name, 'classifier_b': 'MANA',
                             'subset': 'A3', 'p_value': pval,
                             'significant_at_005': pval < 0.05})
        print(f"  {name} vs MANA (A3 only): p={pval:.6f} {'***' if pval < 0.05 else ''}")

    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(RESULTS_DIR, 'mcnemar_results.csv'), index=False)

except ImportError:
    print("  WARNING: statsmodels not installed — skipping McNemar's test.")
    print("  Install with: conda install statsmodels")

# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*70}")
print(f"PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"Results saved to: {RESULTS_DIR}")
print(f"\nFiles generated:")
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"  {f:<45} {size:>10,} bytes")
print(f"\n{'='*70}")