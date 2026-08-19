#!/usr/bin/env python
"""Inject spoofing attacks into REAL per-epoch series (capture semantics).

Why this exists: real vessel logs give only NEGATIVES (no labeled spoofing),
so recall on real traffic is unmeasurable without injection. Getting the
attack model right matters:

  A naive lat/lon shift applied identically to both receivers PRESERVES the
  inter-receiver geometry, silently disabling the strongest physical signal
  (inter_rx_distance). A real single-transmitter spoofer CAPTURES both
  receivers: they lock onto the same counterfeit signal, so their reported
  positions collapse to (nearly) the same point. This injector models
  capture explicitly and leaves SNR untouched (we cannot synthesize
  believable SNR, and the deployable feature profile excludes it anyway).

Attack model (per window):
  1. capture at onset t0: for t >= t0, rx2 position := rx1 position +
     N(0, capture_sigma_m)  (baseline collapse)
  2. drift: both receivers' positions shift by
     offset(t) = drift_mps * (t - t0) along bearing theta
  3. replay: constant offset_m along theta instead of growing drift
  Reported SOG/COG are left as-is (the mismatch between reported kinematics
  and position-derived kinematics is itself a real spoofing signature that
  the derived features pick up).

Operates on the tidy epoch frame produced by mspoof.epochs.epochs_to_frame
(columns: receiver, utc_time, lat_deg, lon_deg, ...). Derived features must
be RECOMPUTED after injection (this script re-splits the frame and calls the
normal feature path downstream), never copied from the clean series.

Usage (library):
    from scripts.inject_attacks import inject_frame
    attacked = inject_frame(frame, kind='drift', drift_mps=0.5,
                            bearing_deg=45.0, onset_frac=0.5, seed=0)

Self-test:
    python scripts/inject_attacks.py --selftest
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mspoof import geo  # noqa: E402

M_PER_DEG_LAT = 111_320.0


def _offset_latlon(lat, lon, dist_m, bearing_deg):
    """First-order local offset; exact enough for <= km-scale spoof offsets."""
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / M_PER_DEG_LAT
    dlon = (dist_m * math.sin(b)) / (M_PER_DEG_LAT *
                                     max(math.cos(math.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon


def inject_frame(frame, kind='drift', drift_mps=0.5, offset_m=50.0,
                 bearing_deg=45.0, onset_frac=0.5, capture_sigma_m=0.5,
                 seed=0):
    """Return an attacked copy of a tidy epoch frame.

    kind: 'drift' (gradual, MARSIM A3-like) or 'replay' (constant offset,
    A1/A2-like). onset_frac: attack begins at this fraction of the window,
    so training/evaluation windows include PARTIAL-onset cases, which
    full-recording MARSIM aggregates never contain.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    t = out['utc_time'].to_numpy(float)
    t0 = t.min() + onset_frac * (t.max() - t.min())
    active = t >= t0

    # 1) capture: rx2 collapses onto rx1's reported track
    rx1 = out['receiver'].to_numpy() == 1
    rx2 = ~rx1
    r1 = out[rx1].set_index('utc_time')
    for idx in out.index[rx2 & active]:
        tt = out.at[idx, 'utc_time']
        if tt in r1.index:
            base_lat = r1.at[tt, 'lat_deg']
            out.at[idx, 'lat_deg'] = base_lat + \
                rng.normal(0, capture_sigma_m) / M_PER_DEG_LAT
            out.at[idx, 'lon_deg'] = r1.at[tt, 'lon_deg'] + \
                rng.normal(0, capture_sigma_m) / (
                    M_PER_DEG_LAT * max(math.cos(math.radians(base_lat)), 1e-6))

    # 2) common spoofed displacement applied to BOTH receivers
    for idx in out.index[active]:
        la, lo = out.at[idx, 'lat_deg'], out.at[idx, 'lon_deg']
        if np.isnan(la) or np.isnan(lo):
            continue
        if kind == 'drift':
            d = drift_mps * (out.at[idx, 'utc_time'] - t0)
        elif kind == 'replay':
            d = offset_m
        else:
            raise ValueError(f'unknown attack kind {kind!r}')
        out.at[idx, 'lat_deg'], out.at[idx, 'lon_deg'] = \
            _offset_latlon(la, lo, d, bearing_deg)

    out.attrs['attack'] = {'kind': kind, 'onset_utc': float(t0),
                           'drift_mps': drift_mps, 'offset_m': offset_m,
                           'bearing_deg': bearing_deg, 'seed': seed}
    return out


def _selftest():
    # synthetic straight north track, two receivers 4 m apart, 120 epochs
    rows = []
    for i in range(120):
        lat = 40.0 + i * (3.0 / M_PER_DEG_LAT)   # ~3 m/s north
        for rx, dlon in ((1, 0.0), (2, 4.0 / (M_PER_DEG_LAT *
                                              math.cos(math.radians(40.0))))):
            rows.append({'receiver': rx, 'utc_time': 1000.0 + i,
                         'lat_deg': lat, 'lon_deg': -70.0 + dlon})
    f = pd.DataFrame(rows)

    a = inject_frame(f, kind='drift', drift_mps=0.5, onset_frac=0.5, seed=1)

    def inter_rx_dist(fr, t):
        g = fr[fr.utc_time == t]
        p1 = g[g.receiver == 1].iloc[0]
        p2 = g[g.receiver == 2].iloc[0]
        return geo.haversine_m(p1.lat_deg, p1.lon_deg, p2.lat_deg, p2.lon_deg)

    pre = inter_rx_dist(a, 1010.0)
    post = inter_rx_dist(a, 1110.0)
    assert 3.5 < pre < 4.5, f'pre-onset baseline broken: {pre}'
    assert post < 2.0, f'capture collapse missing: {post}'

    # drift accumulates on rx1 vs the clean track
    clean = f[(f.receiver == 1) & (f.utc_time == 1119.0)].iloc[0]
    att = a[(a.receiver == 1) & (a.utc_time == 1119.0)].iloc[0]
    disp = geo.haversine_m(clean.lat_deg, clean.lon_deg,
                           att.lat_deg, att.lon_deg)
    expect = 0.5 * (1119.0 - a.attrs['attack']['onset_utc'])
    assert abs(disp - expect) < 2.0, (disp, expect)
    print(f'selftest OK (pre {pre:.2f} m, post-capture {post:.2f} m, '
          f'drift {disp:.1f} m ~= {expect:.1f} m)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print('Use as a library (inject_frame) or run --selftest.')
