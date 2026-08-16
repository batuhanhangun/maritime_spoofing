#!/usr/bin/env python
"""Extract per-recording features (and per-epoch series) from MARSIM pcaps.

Changes vs the original extract_features.py:
* geodesically correct position-derived heading (mspoof.geo.bearing_deg)
* talker-agnostic sentence matching
* optional per-epoch parquet dump (enables LSTM/Transformer + windowed eval)
* multiprocessing (the original was single-process)
* config-file paths instead of hardcoded Windows paths
* feature profile switch: full | deployable

Usage:
    python scripts/extract_features.py --config config.yaml
    python scripts/extract_features.py --config config.yaml --profile deployable
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mspoof.config import load_config, resolve_workers  # noqa: E402
from mspoof.readers.pcap_reader import read_pcap  # noqa: E402
from mspoof.epochs import build_epoch_list, epochs_to_frame  # noqa: E402
from mspoof.features import extract_features  # noqa: E402
from mspoof.nmea import safe_float  # noqa: E402

_WORKER_STATE = {}


def _init_worker(dataset_dir, receivers, profile, epochs_dir, validate_cs):
    _WORKER_STATE.update(dataset_dir=dataset_dir, receivers=receivers,
                         profile=profile, epochs_dir=epochs_dir,
                         validate_cs=validate_cs)


def _meta_row(entry):
    row = {'filename': entry['filename'], 'scenario': entry['scenario'],
           'label': entry['label'], 'index': int(entry['index'])}
    params = entry.get('parameters', {})
    for slot, (k, v) in enumerate(params.items(), start=1):
        row[f'param_{slot}_name'] = k
        row[f'param_{slot}_value'] = safe_float(v)
    return row


def _process_entry(entry):
    st = _WORKER_STATE
    row = _meta_row(entry)
    filepath = os.path.join(st['dataset_dir'], entry['filename'])
    try:
        rx1_s, rx2_s, rx1_g, rx2_g = read_pcap(
            filepath, receivers=st['receivers'],
            validate_checksums=st['validate_cs'])
        rx1 = build_epoch_list(rx1_s, rx1_g)
        rx2 = build_epoch_list(rx2_s, rx2_g)

        row.update(extract_features(rx1, rx2, profile=st['profile']))

        if st['epochs_dir']:
            frame = epochs_to_frame(rx1, rx2)
            out_dir = os.path.join(st['epochs_dir'], entry['scenario'])
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(entry['filename']))[0]
            frame.to_parquet(os.path.join(out_dir, base + '.parquet'), index=False)
        row['_error'] = ''
    except Exception as exc:  # keep the row; surface the error
        row['_error'] = str(exc)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--dataset-dir', default=None)
    ap.add_argument('--profile', default=None, choices=['full', 'deployable'])
    ap.add_argument('--no-epoch-series', action='store_true',
                    help='skip the per-epoch parquet dump')
    args = ap.parse_args()

    cfg = load_config(args.config, **{
        'paths.dataset_dir': args.dataset_dir,
        'features.profile': args.profile,
    })

    dataset_dir = cfg.paths.dataset_dir
    with open(os.path.join(dataset_dir, 'dataset.json')) as fh:
        dataset = json.load(fh)

    receivers = {tuple(cfg.receivers.rx1): 1, tuple(cfg.receivers.rx2): 2}
    epochs_dir = None if args.no_epoch_series or not cfg.extraction.save_epoch_series \
        else cfg.paths.epochs_dir
    profile = cfg.features.profile
    n_workers = resolve_workers(cfg.extraction.n_workers)

    print(f'MARSIM extraction: {len(dataset)} files | profile={profile} | '
          f'workers={n_workers} | epoch series -> {epochs_dir or "OFF"}')

    results = []
    t0 = time.time()
    with multiprocessing.Pool(
            processes=n_workers, initializer=_init_worker,
            initargs=(dataset_dir, receivers, profile, epochs_dir,
                      cfg.extraction.validate_checksums)) as pool:
        for i, row in enumerate(pool.imap_unordered(_process_entry, dataset, chunksize=20)):
            results.append(row)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(dataset) - i - 1) / rate
                print(f'  [{i + 1:>6}/{len(dataset)}] {rate:.1f} files/s, ETA {eta / 60:.0f} min')

    df = pd.DataFrame(results)
    n_err = int((df['_error'] != '').sum())
    print(f'Done in {(time.time() - t0) / 60:.1f} min | errors: {n_err}')
    if n_err:
        print(df.loc[df['_error'] != '', ['filename', '_error']].head(10).to_string())

    meta_cols = ['filename', 'scenario', 'label', 'index',
                 'param_1_name', 'param_1_value', 'param_2_name', 'param_2_value']
    feature_cols = sorted(c for c in df.columns if c not in meta_cols + ['_error'])
    df = df[[c for c in meta_cols if c in df.columns] + feature_cols + ['_error']]

    os.makedirs(os.path.dirname(cfg.paths.features_csv), exist_ok=True)
    df.to_csv(cfg.paths.features_csv, index=False)
    print(f'Features ({len(feature_cols)} cols) -> {cfg.paths.features_csv}')
    print(f"Label balance: {df['label'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()
