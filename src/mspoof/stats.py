"""Statistical helpers for honest reporting.

Motivated by two flaws found in the CIMSS-era analysis:
1. Precision-at-prevalence was propagated from a POINT estimate of FPR
   measured on only ~4.3k negatives. The rule-of-three resolution floor
   (~3/n ~= 7e-4) is ABOVE the prevalences of interest (1e-3, 1e-4), and a
   zero-FP classifier yields precision == 1.0 at any prevalence, which is
   statistically meaningless. All rate estimates therefore carry
   Clopper-Pearson intervals here.
"""
import numpy as np
from scipy.stats import beta


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def precision_at_prevalence(tpr, fpr, pi):
    """precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))."""
    denom = tpr * pi + fpr * (1 - pi)
    return (tpr * pi / denom) if denom > 0 else np.nan


def precision_at_prevalence_interval(tp, fn, fp, tn, pi, alpha=0.05):
    """Conservative interval for precision at prevalence pi.

    Uses the pessimistic corner (TPR lower bound, FPR upper bound) and the
    optimistic corner (TPR upper, FPR lower). The point estimate uses the
    observed rates; when fp == 0 the upper FPR bound (rule-of-three-like)
    is what keeps the optimistic precision below 1.0.
    """
    n_pos, n_neg = tp + fn, fp + tn
    tpr = tp / n_pos if n_pos else np.nan
    fpr = fp / n_neg if n_neg else np.nan
    tpr_lo, tpr_hi = clopper_pearson(tp, n_pos, alpha)
    fpr_lo, fpr_hi = clopper_pearson(fp, n_neg, alpha)
    return {
        'point': precision_at_prevalence(tpr, fpr, pi),
        'lower': precision_at_prevalence(tpr_lo, fpr_hi, pi),
        'upper': precision_at_prevalence(tpr_hi, fpr_lo, pi),
        'tpr': tpr, 'fpr': fpr,
        'tpr_ci': (tpr_lo, tpr_hi), 'fpr_ci': (fpr_lo, fpr_hi),
    }
