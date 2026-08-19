"""Shared preprocessing pieces.

iterative_corr_filter implements the algorithm the CIMSS paper TEXT describes
("among features with |r| > 0.95, iteratively remove the feature with the
highest mean absolute correlation to its remaining peers"). The previous code
silently used a different rule (greedy upper-triangle drop keeping whichever
correlated feature sorts first), so paper and code disagreed. The journal
version uses this implementation everywhere; the old rule is gone.
"""
import numpy as np


def iterative_corr_filter(X_train, threshold=0.95):
    """Return list of columns to DROP.

    While any pair exceeds `threshold`, remove the feature with the highest
    mean absolute correlation to the remaining features. Deterministic:
    ties break by column name.
    """
    corr = X_train.corr().abs()
    cols = list(corr.columns)
    dropped = []
    while True:
        vals = corr.loc[cols, cols].to_numpy(copy=True)
        np.fill_diagonal(vals, 0.0)
        import pandas as pd
        sub = pd.DataFrame(vals, index=cols, columns=cols)
        if not (sub.values > threshold).any():
            break
        mean_abs = sub.mean(axis=1)
        # restrict to columns involved in at least one offending pair
        involved = sub.index[(sub > threshold).any(axis=1)]
        victim = sorted(involved, key=lambda c: (-mean_abs[c], c))[0]
        dropped.append(victim)
        cols.remove(victim)
    return dropped
