"""Aggregation of per-epoch series into per-recording feature vectors.

Two feature profiles are defined:

* ``full``: all 138 features from the CIMSS paper (18 raw + 6 derived + 3
  multi-receiver keys, times 5 summary stats, plus clock_regularity and two
  epoch counts).
* ``deployable``: only features computable from the sentence types that real
  vessels actually log (GGA + VTG + ZDA, verified against NOAA OMAO-SCS and
  R2R fleet holdings). Drops everything sourced from GSV (SNR statistics) and
  GSA (PDOP/VDOP/HDOP_gsa/fix mode). RMC-sourced sog/cog survive because the
  canonicalization in epochs.py falls back to VTG.

The profile mechanism is what lets the journal paper report the "does the
MARSIM advantage survive on real bridge equipment" experiment with a single
flag instead of a forked pipeline.
"""

import math

import numpy as np

from .epochs import (RAW_EPOCH_KEYS, DERIVED_KEYS, INTER_RX_KEYS,
                     compute_derived, compute_inter_receiver)

STATS = ('mean', 'std', 'min', 'max', 'median')

_STAT_FUNCS = {
    'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max,
    'median': np.median,
}

# Keys unavailable on real vessel logs (no GSV/GSA sentences logged).
_NON_DEPLOYABLE_KEYS = {
    'mean_snr', 'std_snr', 'min_snr', 'max_snr', 'num_sats_with_snr',
    'pdop', 'hdop_gsa', 'vdop', 'fix_mode',
}

PROFILES = {
    'full': RAW_EPOCH_KEYS,
    'deployable': [k for k in RAW_EPOCH_KEYS if k not in _NON_DEPLOYABLE_KEYS],
}


def _safe_stat(values, func):
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float('nan')
    return float(func(clean))


def aggregate_series(values, prefix):
    return {f'{prefix}_{s}': _safe_stat(values, _STAT_FUNCS[s]) for s in STATS}


def extract_features(rx1_epochs, rx2_epochs, profile='full'):
    """Per-recording (or per-window) feature vector from two epoch lists."""
    if profile not in PROFILES:
        raise ValueError(f'unknown profile {profile!r}; choose from {sorted(PROFILES)}')
    raw_keys = PROFILES[profile]

    features = {}
    primary = rx1_epochs if rx1_epochs else rx2_epochs
    if not primary:
        return features

    for key in raw_keys:
        values = [e.get(key, float('nan')) for e in primary]
        features.update(aggregate_series(values, key))

    derived = compute_derived(primary)
    for key in DERIVED_KEYS:
        features.update(aggregate_series(derived.get(key, []), key))

    time_deltas = [v for v in derived.get('time_delta', [])
                   if not (isinstance(v, float) and math.isnan(v))]
    features['clock_regularity'] = float(np.std(time_deltas)) if time_deltas else float('nan')

    inter = compute_inter_receiver(rx1_epochs, rx2_epochs)
    for key in INTER_RX_KEYS:
        features.update(aggregate_series(inter.get(key, []), key))

    features['n_epochs_rx1'] = len(rx1_epochs)
    features['n_epochs_rx2'] = len(rx2_epochs)
    return features


def feature_names(profile='full'):
    """Deterministic list of feature column names for a profile."""
    names = []
    for key in PROFILES[profile]:
        names += [f'{key}_{s}' for s in STATS]
    for key in DERIVED_KEYS:
        names += [f'{key}_{s}' for s in STATS]
    names.append('clock_regularity')
    for key in INTER_RX_KEYS:
        names += [f'{key}_{s}' for s in STATS]
    names += ['n_epochs_rx1', 'n_epochs_rx2']
    return names
