"""Train/test split regimes for the MARSIM feature table.

All functions take the full feature DataFrame (with metadata columns
``scenario``, ``label``, ``index``, ``param_1_name``, ``param_1_value``,
``param_2_name``, ``param_2_value``) and return boolean masks
(train_mask, test_mask), so callers never duplicate slicing logic.

Regimes:

* ``repetition_split``: the CIMSS split. Repetition indices 0-15 train,
  16-19 test. Same parameter combinations appear on both sides; only the
  stochastic repetition differs.
* ``kfold_repetition``: 5 folds over repetition indices (0-3, 4-7, ...,
  16-19 as test in turn) for mean-and-std reporting, addressing the
  single-split limitation.
* ``param_holdout``: the generalization experiment reviewers asked for.
  Entire regions of a parameter axis are held out: models never see ANY
  repetition of the held-out parameter values at training time. Example:
  train on A3 shift_speed <= 48, test on shift_speed >= 52 (extrapolation),
  or hold out interior bands (interpolation).

MARSIM detail that matters here: unspoofed recordings carry the same
parameter annotations as their spoofed counterparts (they are the paired
baselines generated per parameter combination), so a parameter holdout
cleanly moves both classes together and preserves the 50/50 balance.
"""

import numpy as np
import pandas as pd

REPETITION_TRAIN = tuple(range(0, 16))
REPETITION_TEST = tuple(range(16, 20))


def repetition_split(df, train_indices=REPETITION_TRAIN, test_indices=REPETITION_TEST):
    idx = df['index'].astype(int)
    return idx.isin(train_indices).values, idx.isin(test_indices).values


def kfold_repetition(df, n_folds=5):
    """Yield (fold_id, train_mask, test_mask) with contiguous repetition folds."""
    idx = df['index'].astype(int)
    all_reps = np.arange(20)
    fold_size = 20 // n_folds
    for fold in range(n_folds):
        test_reps = set(all_reps[fold * fold_size:(fold + 1) * fold_size].tolist())
        train_reps = set(all_reps.tolist()) - test_reps
        yield fold, idx.isin(train_reps).values, idx.isin(test_reps).values


def _param_values(df, param_name):
    """Return a Series of the requested parameter's value per row (NaN where
    the row does not carry that parameter)."""
    vals = pd.Series(np.nan, index=df.index)
    for slot in ('1', '2'):
        mask = df[f'param_{slot}_name'] == param_name
        vals.loc[mask] = df.loc[mask, f'param_{slot}_value'].values
    return vals


def param_holdout(df, scenario, param_name, test_values=None, test_min=None,
                  test_max=None, other_scenarios='exclude'):
    """Hold out parameter regions of one scenario.

    Exactly one of ``test_values`` (explicit iterable) or a
    ``test_min``/``test_max`` range must describe the held-out test region.
    Rows of the target scenario whose parameter falls in the region go to
    test; the remaining rows of that scenario go to train.

    other_scenarios:
        'exclude' (default): A1/A2 rows are dropped entirely, giving a clean
            single-scenario generalization experiment.
        'train': A1/A2 rows are added to the training side (tests whether
            cross-attack data helps generalization to unseen A3 regions).
    """
    if (test_values is None) == (test_min is None and test_max is None):
        raise ValueError('specify test_values OR a test_min/test_max range')

    in_scenario = (df['scenario'] == scenario).values
    vals = _param_values(df, param_name)

    if test_values is not None:
        in_region = vals.isin(list(test_values)).values
    else:
        lo = -np.inf if test_min is None else test_min
        hi = np.inf if test_max is None else test_max
        in_region = ((vals >= lo) & (vals <= hi)).values

    test_mask = in_scenario & in_region
    train_mask = in_scenario & ~in_region & ~vals.isna().values

    if other_scenarios == 'train':
        train_mask = train_mask | ~in_scenario
    elif other_scenarios != 'exclude':
        raise ValueError("other_scenarios must be 'exclude' or 'train'")

    return train_mask, test_mask


# Named experiment presets used by the training script and reported in the
# journal paper, so the paper text and the code stay in sync by construction.
PARAM_HOLDOUT_PRESETS = {
    # Extrapolate DOWN into the hardest region: train on fast drift only,
    # test on the slow drift regime where MANA's cliff lives.
    'a3_speed_low': dict(scenario='A3', param_name='shift_speed', test_max=16),
    # Extrapolate UP: train slow, test fast (sanity direction).
    'a3_speed_high': dict(scenario='A3', param_name='shift_speed', test_min=52),
    # Interpolate an interior band of angles never seen in training.
    'a3_angle_band': dict(scenario='A3', param_name='shift_angle',
                          test_min=60, test_max=120),
    # A2 fine-delay holdout: unseen small delays.
    'a2_delay_low': dict(scenario='A2', param_name='delay', test_max=0.06),
}
