"""Decision-threshold policies for distribution shift.

Empirical findings on MARSIM A3 (extrapolation regime: train shift_speed > 16,
test <= 16, deployable features) that motivate this module:

* default 0.5:  RF recall 0.867, XGB 0.718, LGBM 0.681 (collapse), yet
  test AUC stays 0.98-0.995 -> the failure is the OPERATING POINT, not the
  representation.
* val_max_f1 (textbook practice): WORSE than default (RF 0.690). An
  in-distribution validation set contains only strong/easy attacks, so
  max-F1 pushes the threshold up, exactly wrong for unseen weak attacks.
* negatives-anchored fpr_target (threshold = quantile of validation
  NEGATIVE scores): RF recall 0.971 @ FPR 1.1% (F1 0.980, within 0.002 of
  the test-oracle ceiling). Works because the negative class does not shift
  between regimes; only the attacks do.

Use fpr_target in anything deployment-facing; report the others as baselines.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve


def pick_threshold(scores_val, y_val, policy='fpr_target', target_fpr=0.01):
    """Choose a decision threshold WITHOUT test labels.

    policy:
      'default'    -> 0.5
      'val_max_f1' -> argmax F1 on the validation set (known-bad under
                      downward intensity shift; kept as a baseline)
      'fpr_target' -> (1 - target_fpr) quantile of validation NEGATIVE scores
    """
    scores_val = np.asarray(scores_val, dtype=float)
    y_val = np.asarray(y_val)
    if policy == 'default':
        return 0.5
    if policy == 'val_max_f1':
        p, r, th = precision_recall_curve(y_val, scores_val)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        return float(th[int(np.argmax(f1[:-1]))])
    if policy == 'fpr_target':
        neg = scores_val[y_val == 0]
        if len(neg) == 0:
            return 0.5
        return float(np.quantile(neg, 1.0 - target_fpr))
    raise ValueError(f'unknown policy {policy!r}')


def scores_of(model, X):
    """Continuous scores for ranking/thresholding.

    predict_proba when available, else decision_function (e.g. SVC with
    probability=False, which avoids the hidden 5-fold Platt refit).
    """
    if hasattr(model, 'predict_proba'):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)
