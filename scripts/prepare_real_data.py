#!/usr/bin/env python
"""Phase R2: turn one R2R cruise (two GNSS devices) into evaluation assets.

Outputs, under --out-dir/<cruise_id>/:
  windows.csv          per-window deployable features + metadata columns:
                       cruise_id, window_start, n_epochs_win, transit_tag,
                       sog_med_kn, cog_circ_std_deg
  epochs_rx1.parquet   per-epoch series (for injection / time-to-detection)
  epochs_rx2.parquet
  baseline.json        cruise inter-receiver stats (median/MAD of
                       inter_rx_distance_m over presumed-clean windows),
                       device lag estimates, epoch counts
  ks_audit.csv         (with --marsim-features) per-feature KS statistic
                       between this cruise's windows and MARSIM's unspoofed
                       class, sorted by divergence

Transit tagging: a window is 'steady' when median SOG >= --transit-sog-kn
and the circular std of COG <= --transit-cog-deg. Circular statistics are
used ONLY for this metadata tag; the feature vector itself keeps the exact
MARSIM definitions (including linear COG std) so that MARSIM-trained models
see identically-computed inputs.

Usage (FK180528 example):
  python scripts/prepare_real_data.py --cruise-id FK180528 ^
      --rx1 "C:/.../FK180528_129529_gnss" ^
      --rx2 "C:/.../FK180528_129537_ins" ^
      --out-dir results/real --marsim-features data/marsim_features.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mspoof.readers.r2r_reader import read_r2r_device          # noqa: E402
from mspoof.readers.scs_reader import sliding_windows          # noqa: E402
from mspoof.epochs import build_epoch_list, compute_inter_receiver  # noqa: E402
from mspoof.features import extract_features                   # noqa: E402
from mspoof.ablation import family_of                          # noqa: E402


def circ_std_deg(deg_values):
    v = [d for d in deg_values if not math.isnan(d)]
    if len(v) < 2:
        return float('nan')
    rad = np.radians(v)
    R = math.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    R = min(max(R, 1e-12), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(R)))


def epochs_to_parquet(epochs, path):
    if not epochs:
        return
    pd.DataFrame(epochs).to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cruise-id', required=True)
    ap.add_argument('--rx1', required=True, help='primary device dir (e.g. C-Nav)')
    ap.add_argument('--rx2', default=None, help='second device dir (e.g. Seapath)')
    ap.add_argument('--out-dir', default='results/real')
    ap.add_argument('--window', type=float, default=120.0)
    ap.add_argument('--stride', type=float, default=60.0)
    ap.add_argument('--min-epochs', type=int, default=100)
    ap.add_argument('--transit-sog-kn', type=float, default=3.0)
    ap.add_argument('--transit-cog-deg', type=float, default=5.0)
    ap.add_argument('--marsim-features', default=None)
    ap.add_argument('--no-checksums', action='store_true',
                    help='skip NMEA checksum validation (faster)')
    args = ap.parse_args()

    out = os.path.join(args.out_dir, args.cruise_id)
    os.makedirs(out, exist_ok=True)
    validate = not args.no_checksums

    st1, st2 = {}, {}
    print(f'[{args.cruise_id}] reading rx1: {args.rx1}')
    s1, g1 = read_r2r_device(args.rx1, validate_checksums=validate, stats_out=st1)
    e1 = build_epoch_list(s1, g1)
    e2 = []
    if args.rx2:
        print(f'[{args.cruise_id}] reading rx2: {args.rx2}')
        s2, g2 = read_r2r_device(args.rx2, validate_checksums=validate, stats_out=st2)
        e2 = build_epoch_list(s2, g2)
    print(f'[{args.cruise_id}] epochs rx1={len(e1)} rx2={len(e2)} '
          f'(lag rx1={st1.get("lag_s", float("nan")):.3f}s '
          f'rx2={st2.get("lag_s", float("nan")):.3f}s)')

    epochs_to_parquet(e1, os.path.join(out, 'epochs_rx1.parquet'))
    epochs_to_parquet(e2, os.path.join(out, 'epochs_rx2.parquet'))

    rx2_times = [e['utc_time'] for e in e2]
    rows, j0 = [], 0
    for start, w1 in sliding_windows(e1, args.window, args.stride,
                                     args.min_epochs):
        w2 = []
        if e2:
            while j0 < len(rx2_times) and rx2_times[j0] < start:
                j0 += 1
            j = j0
            while j < len(rx2_times) and rx2_times[j] < start + args.window:
                j += 1
            w2 = e2[j0:j]
        feats = extract_features(w1, w2, profile='deployable')
        sogs = [e.get('sog_knots', float('nan')) for e in w1]
        sog_med = float(np.nanmedian(sogs)) if sogs else float('nan')
        ccs = circ_std_deg([e.get('cog_deg', float('nan')) for e in w1])
        feats.update({
            'cruise_id': args.cruise_id,
            'window_start': start,
            'n_epochs_win': len(w1),
            'sog_med_kn': sog_med,
            'cog_circ_std_deg': ccs,
            'transit_tag': int(sog_med >= args.transit_sog_kn
                               and ccs <= args.transit_cog_deg),
        })
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'windows.csv'), index=False)
    n_transit = int(df.transit_tag.sum()) if len(df) else 0
    print(f'[{args.cruise_id}] windows={len(df)} '
          f'(steady-transit={n_transit}, '
          f'{100.0 * n_transit / max(len(df), 1):.1f}%)')

    # Cruise baseline for inter-receiver centering at evaluation time.
    baseline = {'cruise_id': args.cruise_id, 'n_windows': len(df),
                'rx1_stats': st1, 'rx2_stats': st2}
    if len(df) and 'inter_rx_distance_m_median' in df:
        v = df['inter_rx_distance_m_median'].dropna()
        if len(v):
            med = float(v.median())
            baseline['inter_rx_distance_m'] = {
                'median': med,
                'mad': float((v - med).abs().median()),
                'n': int(len(v)),
            }
    with open(os.path.join(out, 'baseline.json'), 'w') as fh:
        json.dump(baseline, fh, indent=2)
    print(f'[{args.cruise_id}] baseline: '
          f'{baseline.get("inter_rx_distance_m", "single receiver")}')

    # KS audit against MARSIM's unspoofed class (deployable features only).
    if args.marsim_features and len(df):
        from scipy.stats import ks_2samp
        m = pd.read_csv(args.marsim_features)
        m = m[m.label == 'unspoofed']
        recs = []
        for col in df.columns:
            if col not in m.columns or family_of(col) in ('snr', 'gsa'):
                continue
            a = df[col].dropna().values
            b = m[col].dropna().values
            if len(a) < 20 or len(b) < 20:
                continue
            ks = ks_2samp(a, b)
            recs.append({'feature': col, 'ks_stat': ks.statistic,
                         'real_median': float(np.median(a)),
                         'marsim_median': float(np.median(b))})
        if not recs:
            print(f'[{args.cruise_id}] KS audit skipped: too few windows '
                  f'(need >= 20 per feature)')
        else:
            audit = pd.DataFrame(recs).sort_values('ks_stat', ascending=False)
            audit.to_csv(os.path.join(out, 'ks_audit.csv'), index=False)
            print(f'[{args.cruise_id}] KS audit: top divergent features:')
            print(audit.head(8).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
