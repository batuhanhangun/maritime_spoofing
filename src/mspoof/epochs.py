"""Epoch assembly and per-epoch derived features.

An "epoch" is one second of receiver output, keyed by the UTC field of the
anchoring RMC/GGA sentence. Readers (pcap, SCS) produce, per receiver:

    sentences: {utc_seconds: {key: value, ...}}
    gsv:       {utc_seconds: [snr, snr, ...]}

This module turns those into a sorted per-epoch table, canonicalizes
overlapping fields, computes consecutive-epoch derived features, and computes
multi-receiver comparison features. It is reader-agnostic: MARSIM pcaps and
real SCS logs flow through identical code from here on.
"""

import math

import numpy as np
import pandas as pd

from .geo import haversine_m, bearing_deg, angular_diff_deg, MPS_TO_KNOTS

# Per-epoch raw keys aggregated downstream. Order matters only for readability.
RAW_EPOCH_KEYS = [
    'sog_knots', 'cog_deg',                      # canonical speed/course
    'num_satellites', 'hdop', 'altitude',        # GGA
    'pdop', 'hdop_gsa', 'vdop', 'fix_mode',      # GSA
    'mean_snr', 'std_snr', 'min_snr', 'max_snr', 'num_sats_with_snr',  # GSV
    'sog_vtg_knots', 'sog_vtg_kmh', 'cog_true', 'cog_magnetic',        # VTG
]

DERIVED_KEYS = [
    'position_jump_m', 'computed_sog_knots', 'sog_discrepancy',
    'cog_change_rate', 'cog_heading_discrepancy', 'time_delta',
]

INTER_RX_KEYS = ['inter_rx_distance_m', 'inter_rx_sog_diff', 'inter_rx_cog_diff']

EPOCH_MATCH_TOL = 0.5  # seconds; max |t1 - t2| for cross-receiver matching


def _canonicalize(epoch):
    """Fill canonical sog/cog from whichever source sentence is available.

    Priority: RMC, then VTG. MARSIM provides both; real SCS logs frequently
    lack RMC entirely (NOAA fleet logs GGA/VTG/ZDA), so falling back to VTG is
    what makes the deployable feature profile computable on real data.
    """
    sog = epoch.get('sog_rmc_knots', float('nan'))
    if math.isnan(sog):
        sog = epoch.get('sog_vtg_knots', float('nan'))
    epoch['sog_knots'] = sog

    cog = epoch.get('cog_rmc_deg', float('nan'))
    if math.isnan(cog):
        cog = epoch.get('cog_true', float('nan'))
    epoch['cog_deg'] = cog
    return epoch


def build_epoch_list(sentences, gsv_data):
    """{utc: {...}} dicts -> sorted list of canonical epoch dicts."""
    epochs = []
    for t in sorted(sentences.keys()):
        epoch = dict(sentences[t])
        epoch['utc_time'] = t

        snrs = gsv_data.get(t, [])
        if snrs:
            epoch['mean_snr'] = float(np.mean(snrs))
            epoch['std_snr'] = float(np.std(snrs))
            epoch['min_snr'] = float(np.min(snrs))
            epoch['max_snr'] = float(np.max(snrs))
            epoch['num_sats_with_snr'] = len(snrs)
        else:
            epoch['mean_snr'] = float('nan')
            epoch['std_snr'] = float('nan')
            epoch['min_snr'] = float('nan')
            epoch['max_snr'] = float('nan')
            epoch['num_sats_with_snr'] = 0

        epochs.append(_canonicalize(epoch))
    return epochs


def compute_derived(epochs):
    """Consecutive-epoch dynamics. One value per epoch pair (n-1 values).

    Uses the exact great-circle bearing (geo.bearing_deg) for the
    position-derived heading. The previous implementation used
    atan2(dlon, dlat), which is only valid near due-north courses.
    """
    derived = {k: [] for k in DERIVED_KEYS}

    for i in range(1, len(epochs)):
        prev, curr = epochs[i - 1], epochs[i]
        lat1, lon1 = prev.get('lat_deg', float('nan')), prev.get('lon_deg', float('nan'))
        lat2, lon2 = curr.get('lat_deg', float('nan')), curr.get('lon_deg', float('nan'))
        t1, t2 = prev.get('utc_time', float('nan')), curr.get('utc_time', float('nan'))

        dt = t2 - t1 if not (math.isnan(t1) or math.isnan(t2)) else float('nan')
        # Midnight rollover on real multi-day logs: utc_time resets to ~0.
        if not math.isnan(dt) and dt < -80000:
            dt += 86400.0
        derived['time_delta'].append(dt)

        jump = haversine_m(lat1, lon1, lat2, lon2)
        derived['position_jump_m'].append(jump)

        if not math.isnan(jump) and not math.isnan(dt) and dt > 0:
            computed_sog = (jump / dt) * MPS_TO_KNOTS
            derived['computed_sog_knots'].append(computed_sog)

            reported_sog = curr.get('sog_knots', float('nan'))
            derived['sog_discrepancy'].append(
                abs(reported_sog - computed_sog) if not math.isnan(reported_sog) else float('nan'))

            heading = bearing_deg(lat1, lon1, lat2, lon2)
            reported_cog = curr.get('cog_deg', float('nan'))
            derived['cog_heading_discrepancy'].append(
                angular_diff_deg(reported_cog, heading) if not math.isnan(reported_cog) else float('nan'))
        else:
            derived['computed_sog_knots'].append(float('nan'))
            derived['sog_discrepancy'].append(float('nan'))
            derived['cog_heading_discrepancy'].append(float('nan'))

        cog1 = prev.get('cog_deg', float('nan'))
        cog2 = curr.get('cog_deg', float('nan'))
        if not (math.isnan(cog1) or math.isnan(cog2)) and not math.isnan(dt) and dt > 0:
            derived['cog_change_rate'].append(angular_diff_deg(cog2, cog1) / dt)
        else:
            derived['cog_change_rate'].append(float('nan'))

    return derived


def compute_inter_receiver(rx1_epochs, rx2_epochs, tol=EPOCH_MATCH_TOL):
    """Epoch-matched cross-receiver comparisons.

    O(n) two-pointer matching over the sorted epoch lists (the old version was
    an O(n^2) scan; irrelevant at 120 epochs, painful on multi-hour real logs).
    """
    inter = {k: [] for k in INTER_RX_KEYS}
    if not rx1_epochs or not rx2_epochs:
        return inter

    rx2_times = [e['utc_time'] for e in rx2_epochs]
    j = 0
    for e1 in rx1_epochs:
        t1 = e1['utc_time']
        while j + 1 < len(rx2_times) and abs(rx2_times[j + 1] - t1) <= abs(rx2_times[j] - t1):
            j += 1
        if abs(rx2_times[j] - t1) > tol:
            for k in INTER_RX_KEYS:
                inter[k].append(float('nan'))
            continue
        e2 = rx2_epochs[j]

        # Motion compensation: real devices sample asynchronously (e.g.
        # C-Nav on integer seconds, Seapath at ~x.55 s), so the matched
        # epochs differ by up to `tol` seconds. At 12 kn a 0.5 s offset is
        # ~3 m of along-track displacement, the same order as the capture
        # signal itself. Dead-reckon rx2 to rx1's epoch using rx2's own
        # velocity before differencing. Exact no-op when dt == 0 (MARSIM)
        # or when rx2 lacks a usable velocity.
        lat2 = e2.get('lat_deg', float('nan'))
        lon2 = e2.get('lon_deg', float('nan'))
        dt = t1 - e2['utc_time']
        sog2 = e2.get('sog_knots', float('nan'))
        cog2 = e2.get('cog_deg', float('nan'))
        if dt != 0.0 and not (math.isnan(sog2) or math.isnan(cog2)
                              or math.isnan(lat2) or math.isnan(lon2)):
            d_m = sog2 * 0.514444 * dt
            lat2 = lat2 + d_m * math.cos(math.radians(cog2)) / 111320.0
            lon2 = lon2 + d_m * math.sin(math.radians(cog2)) / (
                111320.0 * max(math.cos(math.radians(lat2)), 1e-6))

        inter['inter_rx_distance_m'].append(
            haversine_m(e1.get('lat_deg', float('nan')), e1.get('lon_deg', float('nan')),
                        lat2, lon2))

        s1, s2 = e1.get('sog_knots', float('nan')), e2.get('sog_knots', float('nan'))
        inter['inter_rx_sog_diff'].append(
            abs(s1 - s2) if not (math.isnan(s1) or math.isnan(s2)) else float('nan'))

        inter['inter_rx_cog_diff'].append(
            angular_diff_deg(e1.get('cog_deg', float('nan')), e2.get('cog_deg', float('nan'))))

    return inter


def epochs_to_frame(rx1_epochs, rx2_epochs):
    """Assemble the per-epoch time series into a single tidy DataFrame.

    This is the artifact the old pipeline threw away. Persisting it (parquet)
    is what enables sequence models (LSTM/Transformer) and sliding-window
    evaluation without re-parsing pcaps.

    Columns: utc_time, receiver (1|2), all RAW_EPOCH_KEYS present, lat/lon,
    plus derived_* columns aligned to receiver 1 (NaN on the first epoch).
    """
    rows = []
    for rx_id, eps in ((1, rx1_epochs), (2, rx2_epochs)):
        derived = compute_derived(eps)
        for i, e in enumerate(eps):
            row = {'receiver': rx_id, 'utc_time': e['utc_time'],
                   'lat_deg': e.get('lat_deg', float('nan')),
                   'lon_deg': e.get('lon_deg', float('nan'))}
            for k in RAW_EPOCH_KEYS:
                row[k] = e.get(k, float('nan'))
            for k in DERIVED_KEYS:
                row[k] = derived[k][i - 1] if i > 0 else float('nan')
            rows.append(row)
    return pd.DataFrame(rows)
