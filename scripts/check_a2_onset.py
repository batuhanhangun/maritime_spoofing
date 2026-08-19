#!/usr/bin/env python
"""Is A2's course/speed separability sustained jitter (physics) or a
splice transient (artifact)?

Plots per-epoch COG and SOG for a few spoofed vs unspoofed A2 recordings.

Reading the output (results/audit/a2_onset_check.png):
  * red traces jittering around the blue ones for the WHOLE recording
    -> physics: fine-delay replay induces sustained kinematic jitter;
       keep the interpretation as-is in the paper.
  * red identical to blue except a spike at one timestamp
    -> splice transient: A2 separability partially reflects attack-onset
       discontinuities in the simulation; say so in the limitations section.

Usage:
    python scripts/check_a2_onset.py
    python scripts/check_a2_onset.py --dataset D:/path/to/dataset/dataset --n 5
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mspoof.readers.pcap_reader import read_pcap          # noqa: E402
from mspoof.epochs import build_epoch_list, epochs_to_frame  # noqa: E402


def load_frame(path):
    s1, s2, g1, g2 = read_pcap(path)
    return epochs_to_frame(build_epoch_list(s1, g1),
                           build_epoch_list(s2, g2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='dataset/dataset',
                    help='MARSIM root containing the pcap files')
    ap.add_argument('--features', default='data/marsim_features.csv')
    ap.add_argument('--n', type=int, default=3,
                    help='recordings per class to plot')
    ap.add_argument('--out', default='results/audit/a2_onset_check.png')
    args = ap.parse_args()

    meta = pd.read_csv(args.features,
                       usecols=['filename', 'scenario', 'label'])
    a2 = meta[meta.scenario == 'A2']
    picks = pd.concat([a2[a2.label == 'spoofed'].head(args.n),
                       a2[a2.label == 'unspoofed'].head(args.n)])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    n_ok = 0
    for _, r in picks.iterrows():
        path = os.path.join(args.dataset, r.filename)
        if not os.path.exists(path):
            print(f'MISSING: {path}')
            continue
        fr = load_frame(path)
        rx1 = fr[fr.receiver == 1].sort_values('utc_time')
        if rx1.empty:
            print(f'EMPTY rx1: {r.filename}')
            continue
        t = rx1.utc_time - rx1.utc_time.min()
        style = dict(lw=1, alpha=0.8,
                     color='tab:red' if r.label == 'spoofed' else 'tab:blue')
        axes[0].plot(t, rx1.cog_deg, **style)
        axes[1].plot(t, rx1.sog_knots, **style)
        n_ok += 1

    if n_ok == 0:
        raise SystemExit('No recordings plotted. Check --dataset path '
                         '(it must contain the pcap files referenced by '
                         'the "filename" column of the features CSV).')

    axes[0].set_ylabel('COG (deg)')
    axes[1].set_ylabel('SOG (kn)')
    axes[1].set_xlabel('seconds into recording')
    axes[0].set_title(f'A2 per-epoch kinematics, rx1 '
                      f'(red = spoofed, blue = unspoofed, {n_ok} recordings)')
    for ax in axes:
        ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()