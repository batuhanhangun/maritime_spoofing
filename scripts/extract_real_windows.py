#!/usr/bin/env python
"""Turn real vessel NMEA logs (NOAA SCS / R2R) into MARSIM-compatible windows.

Input layout (--real-dir):
    real_nmea/
      <cruise_id>/
        rx1/   log files of the primary GNSS device (any names, any count)
        rx2/   log files of the second device (optional; enables the
               inter-receiver feature family)

Output: one CSV of per-window feature vectors (profile: deployable by
default, since real logs lack GSV/GSA) with columns cruise_id, window_start,
plus the feature columns. Feed this to a MARSIM-trained model to measure
false-alarm behavior on real, presumed-clean traffic, and to compare feature
distributions (KS statistics) against MARSIM's unspoofed class.

Notes:
* Windows default to 120 s to match MARSIM's recording length, sliding by
  60 s. Adjust with --window/--stride.
* For the Falkor pair (C-Nav + Seapath), the antennas have a fixed but
  unpublished separation. The evaluation script should baseline
  inter_rx_distance_m against its own cruise-wide median rather than
  MARSIM's 4 m constant; this script just extracts the raw features.

Usage:
    python scripts/extract_real_windows.py --config config.yaml \
        --real-dir /data/real_nmea --out real_windows.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mspoof.config import load_config  # noqa: E402
from mspoof.readers.scs_reader import read_scs_files, sliding_windows  # noqa: E402
from mspoof.epochs import build_epoch_list  # noqa: E402
from mspoof.features import extract_features  # noqa: E402


def load_device(device_dir):
    files = sorted(f for f in glob.glob(os.path.join(device_dir, '*'))
                   if os.path.isfile(f))
    if not files:
        return []
    sentences, gsv = read_scs_files(files)
    return build_epoch_list(sentences, gsv)


def windows_for_cruise(cruise_dir, cruise_id, window_s, stride_s, min_epochs, profile):
    rx1 = load_device(os.path.join(cruise_dir, 'rx1'))
    rx2 = load_device(os.path.join(cruise_dir, 'rx2'))
    print(f'  {cruise_id}: rx1 epochs={len(rx1)}, rx2 epochs={len(rx2)}')
    if not rx1:
        return []

    # Index rx2 windows by aligned start times via a rolling pointer.
    rx2_times = [e['utc_time'] for e in rx2]
    rows = []
    j0 = 0
    for start, w1 in sliding_windows(rx1, window_s, stride_s, min_epochs):
        w2 = []
        if rx2:
            while j0 < len(rx2_times) and rx2_times[j0] < start:
                j0 += 1
            j = j0
            while j < len(rx2_times) and rx2_times[j] < start + window_s:
                j += 1
            w2 = rx2[j0:j]
        feats = extract_features(w1, w2, profile=profile)
        feats['cruise_id'] = cruise_id
        feats['window_start'] = start
        rows.append(feats)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--real-dir', default=None)
    ap.add_argument('--out', default='real_windows.csv')
    ap.add_argument('--window', type=float, default=None)
    ap.add_argument('--stride', type=float, default=None)
    ap.add_argument('--profile', default='deployable', choices=['full', 'deployable'])
    args = ap.parse_args()

    cfg = load_config(args.config, **{
        'paths.real_data_dir': args.real_dir,
        'real_data.window_s': args.window,
        'real_data.stride_s': args.stride,
    })

    real_dir = cfg.paths.real_data_dir
    window_s = cfg.real_data.window_s
    stride_s = cfg.real_data.stride_s
    min_epochs = cfg.real_data.min_epochs

    cruise_dirs = sorted(d for d in glob.glob(os.path.join(real_dir, '*'))
                         if os.path.isdir(d))
    print(f'Real-data extraction: {len(cruise_dirs)} cruise dirs | '
          f'window={window_s}s stride={stride_s}s profile={args.profile}')

    all_rows = []
    for d in cruise_dirs:
        all_rows += windows_for_cruise(d, os.path.basename(d), window_s,
                                       stride_s, min_epochs, args.profile)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f'{len(df)} windows -> {args.out}')


if __name__ == '__main__':
    main()
