"""Reader for R2R academic-fleet GNSS/INS filesets (rvdata.us).

R2R raw filesets differ from interleaved SCS streams in one structural way:
each NMEA sentence family is logged to its OWN file, one file per day, named

    COM18-CNAV-GGA-RAW_20180529-000001.raw
    COM12-Seapath-RMC-RAW_20180529-000001.Raw

with each line carrying an SCS external-clock prefix::

    06/16/2018,00:00:01.818,$GNGGA,000001.50,3111.923871,N,...*6A

Consequences handled here (and NOT handled by scs_reader.read_scs_files,
which assumes sentence families are interleaved in arrival order):

* **Non-anchor attachment.** VTG carries no UTC field; in an interleaved
  stream it attaches to the most recent GGA/RMC anchor. With per-family
  files that heuristic attaches a whole day of VTG to one stale epoch.
  Here VTG lines are re-attached by external clock: the per-device logging
  latency (median of ext_clock - nmea_utc over anchor lines) is estimated,
  then each VTG line is snapped to the nearest anchor epoch within
  ``attach_tol`` seconds.
* **Duplicate copies.** Some fileset downloads contain byte-identical
  ``*_2.raw`` siblings; only one copy per (family, timestamp) is read.
* **Mixed logging rates.** The C-Nav logs at 2 Hz, the Seapath at 1 Hz with
  a ~0.55 s phase. MARSIM-trained models expect ~1 Hz epochs (time_delta
  features), so epochs are optionally decimated to one per integer-second
  bucket (phase-agnostic: the earliest epoch in each bucket is kept).
* **Family filter.** Only GGA/RMC/VTG are read. GLL is redundant with GGA;
  ZDA/HDT/ROT are not used by the feature extractor. GSV/GSA never appear
  in these filesets, which is consistent with the 'deployable' profile.
"""

import math
import os
import re
from bisect import bisect_left
from collections import defaultdict
from statistics import median

from .. import nmea
from .scs_reader import split_log_line

ANCHOR_FAMILIES = ('GGA', 'RMC')
ATTACH_FAMILIES = ('VTG',)
DEFAULT_FAMILIES = ANCHOR_FAMILIES + ATTACH_FAMILIES

# COM18-CNAV-GGA-RAW_20180529-000001.raw  /  ..._2.raw (duplicate copy)
FNAME_RE = re.compile(
    r'^(?P<port>COM\d+)-(?P<device>[A-Za-z0-9]+)-(?P<family>[A-Z]{3})-RAW_'
    r'(?P<stamp>\d{8}-\d{6})(?P<dup>_\d+)?\.raw$', re.IGNORECASE)


def discover_device_files(device_dir, families=DEFAULT_FAMILIES):
    """Map family -> ordered list of file paths, ignoring duplicate copies.

    Duplicate policy: for each (family, stamp) the copy WITHOUT a ``_N``
    suffix wins; a suffixed copy is used only when no unsuffixed one exists.
    """
    best = {}
    for root, _dirs, files in os.walk(device_dir):
        for fn in files:
            m = FNAME_RE.match(fn)
            if not m or m.group('family').upper() not in families:
                continue
            key = (m.group('family').upper(), m.group('stamp'))
            path = os.path.join(root, fn)
            is_dup = m.group('dup') is not None
            if key not in best or (best[key][0] and not is_dup):
                best[key] = (is_dup, path)
    out = defaultdict(list)
    for (family, _stamp), (_dup, path) in sorted(best.items()):
        out[family].append(path)
    return dict(out)


def _iter_lines(paths, validate_checksums=True):
    for path in paths:
        with open(path, 'r', errors='ignore') as fh:
            for line in fh:
                ext_clock, payload = split_log_line(line)
                if payload is None:
                    continue
                if validate_checksums and not nmea.verify_checksum(payload):
                    continue
                fmt = nmea.formatter_of(payload)
                if fmt is None:
                    continue
                yield ext_clock, fmt, nmea.split_sentence(payload)


def _decimate_1hz(sentences):
    """Keep one epoch per integer-second bucket (earliest wins)."""
    kept = {}
    for t in sorted(sentences):
        bucket = math.floor(t)
        if bucket not in kept:
            kept[bucket] = t
    return {t: sentences[t] for t in kept.values()}


def read_r2r_device(device_dir, validate_checksums=True, decimate_1hz=True,
                    attach_tol=0.6, families=DEFAULT_FAMILIES, stats_out=None):
    """Parse one device's R2R fileset into a (sentences, gsv) pair.

    Returns the same structures as scs_reader.read_scs_files, so the result
    feeds directly into mspoof.epochs.build_epoch_list.
    """
    by_family = discover_device_files(device_dir, families)
    sentences = defaultdict(dict)
    offsets = []  # ext_clock - absolute nmea utc, i.e. logging latency

    # Pass 1: anchor families establish the epoch grid.
    from .scs_reader import _absolute_utc
    for family in ANCHOR_FAMILIES:
        for ext_clock, fmt, fields in _iter_lines(by_family.get(family, []),
                                                  validate_checksums):
            if fmt not in nmea.EPOCH_ANCHORS:
                continue
            parsed = nmea.PARSERS[fmt](fields)
            t_rel = parsed.get('utc_time', float('nan'))
            if math.isnan(t_rel):
                continue
            t_abs = _absolute_utc(t_rel, ext_clock)
            parsed['utc_time'] = t_abs
            sentences[t_abs].update(parsed)
            if ext_clock is not None and not math.isnan(ext_clock):
                offsets.append(ext_clock - t_abs)

    if not sentences:
        return {}, {}

    if decimate_1hz:
        sentences = _decimate_1hz(sentences)

    # Pass 2: clock-attached families snap to the nearest surviving epoch.
    lag = median(offsets) if offsets else 0.0
    keys = sorted(sentences)
    n_attached = n_dropped = 0
    for family in ATTACH_FAMILIES:
        for ext_clock, fmt, fields in _iter_lines(by_family.get(family, []),
                                                  validate_checksums):
            if fmt not in nmea.PARSERS or ext_clock is None \
                    or math.isnan(ext_clock):
                n_dropped += 1
                continue
            target = ext_clock - lag
            i = bisect_left(keys, target)
            cands = [k for k in (keys[i - 1] if i else None,
                                 keys[i] if i < len(keys) else None)
                     if k is not None]
            if not cands:
                n_dropped += 1
                continue
            k = min(cands, key=lambda c: abs(c - target))
            if abs(k - target) <= attach_tol:
                # Never let an attached family overwrite anchor fields.
                parsed = nmea.PARSERS[fmt](fields)
                for field, value in parsed.items():
                    sentences[k].setdefault(field, value)
                n_attached += 1
            else:
                n_dropped += 1

    if stats_out is not None:
        stats_out.update({'lag_s': lag, 'attached': n_attached,
                          'attach_dropped': n_dropped,
                          'n_epochs': len(sentences)})
    return dict(sentences), {}


def read_r2r_pair(rx1_dir, rx2_dir, **kwargs):
    """Convenience: read two devices, return (rx1, rx2) epoch-dict pairs."""
    s1, _ = read_r2r_device(rx1_dir, **kwargs)
    s2, _ = read_r2r_device(rx2_dir, **kwargs)
    return s1, s2
