"""End-to-end smoke tests with synthetic data (no MARSIM download needed)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scapy.all import Ether, IP, UDP, Raw, wrpcap

from mspoof.geo import bearing_deg, haversine_m, angular_diff_deg
from mspoof import nmea
from mspoof.readers.pcap_reader import read_pcap
from mspoof.readers.scs_reader import split_log_line, read_scs_files, sliding_windows
from mspoof.epochs import build_epoch_list, epochs_to_frame
from mspoof.features import extract_features, feature_names, PROFILES

TMP = os.path.join(os.path.dirname(__file__), '_tmp')
os.makedirs(TMP, exist_ok=True)


def checksum(body):
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f'${body}*{cs:02X}'


def deg_to_nmea(lat, lon):
    def enc(v, width):
        d = int(abs(v))
        m = (abs(v) - d) * 60
        return f'{d:0{width}d}{m:07.4f}'
    return (enc(lat, 2), 'N' if lat >= 0 else 'S',
            enc(lon, 3), 'E' if lon >= 0 else 'W')


def make_sentences(t, lat, lon, sog=10.0, cog=45.0):
    hh, mm, ss = int(t // 3600), int((t % 3600) // 60), t % 60
    utc = f'{hh:02d}{mm:02d}{ss:05.2f}'
    la, ns, lo, ew = deg_to_nmea(lat, lon)
    return [
        checksum(f'GPRMC,{utc},A,{la},{ns},{lo},{ew},{sog:.1f},{cog:.1f},160826,,,A'),
        checksum(f'GPGGA,{utc},{la},{ns},{lo},{ew},1,08,0.9,1.2,M,0.0,M,,'),
        checksum('GPGSA,A,3,01,02,03,04,05,06,07,08,,,,,1.5,0.9,1.2'),
        checksum('GPGSV,2,1,08,01,40,083,41,02,17,308,43,03,07,344,39,04,22,228,45'),
        checksum('GPGSV,2,2,08,05,13,291,42,06,25,170,40,07,57,208,44,08,67,296,41'),
        checksum(f'GPVTG,{cog:.1f},T,,M,{sog:.1f},N,{sog * 1.852:.1f},K,A'),
    ]


def test_geo():
    # Due-east course at 40N: old atan2(dlon, dlat) would give ~90 only by luck
    # of the argument order; test a NE course where cos(lat) matters.
    lat1, lon1 = 40.0, 29.0
    # ~100 m north and ~100 m east
    lat2 = lat1 + 100.0 / 111320.0
    lon2 = lon1 + 100.0 / (111320.0 * math.cos(math.radians(lat1)))
    b = bearing_deg(lat1, lon1, lat2, lon2)
    assert abs(b - 45.0) < 0.5, f'bearing {b} != 45'
    # Old buggy formula for comparison: atan2(dlon, dlat) without cos scaling
    buggy = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)) % 360
    assert abs(buggy - 45.0) > 5.0, 'bug reproduction check failed'
    d = haversine_m(lat1, lon1, lat2, lon2)
    assert abs(d - 141.4) < 2.0
    assert angular_diff_deg(359.0, 1.0) == 2.0
    print(f'geo OK (bearing={b:.2f}, old-formula bearing={buggy:.2f}, dist={d:.1f} m)')


def test_nmea():
    s = checksum('GPGGA,120001.00,4055.0000,N,02900.0000,E,1,08,0.9,1.2,M,0.0,M,,')
    assert nmea.formatter_of(s) == 'GGA'
    assert nmea.verify_checksum(s)
    assert not nmea.verify_checksum(s[:-1] + ('0' if s[-1] != '0' else '1'))
    assert nmea.formatter_of('$GNRMC,120001,A,4055.0,N,02900.0,E,10.0,45.0,160826,,,A') == 'RMC'
    assert nmea.formatter_of('$PGRMZ,123,f,3') is None
    f = nmea.parse_gga(nmea.split_sentence(s))
    assert abs(f['lat_deg'] - 40.9166666) < 1e-6
    assert f['num_satellites'] == 8
    print('nmea OK')


def test_pcap_pipeline():
    pkts = []
    lat, lon = 40.9, 28.9
    for i in range(130):
        t = 12 * 3600 + i
        # rx1: truth. rx2: 4 m east of rx1, plus a slow drift (spoof-like).
        lat1, lon1 = lat + i * 4e-5, lon + i * 4e-5
        dlon_4m = 4.0 / (111320.0 * math.cos(math.radians(lat1)))
        drift = i * 0.05 / 111320.0  # 5 cm/s drift
        for rx, (la, lo) in ((1, (lat1, lon1)),
                             (2, (lat1 + drift, lon1 + dlon_4m))):
            ip = '192.168.0.10' if rx == 1 else '192.168.0.11'
            port = 62996 if rx == 1 else 62997
            for s in make_sentences(t, la, lo):
                pkts.append(Ether() / IP(src=ip, dst='192.168.0.1')
                            / UDP(sport=port, dport=10110) / Raw(load=s.encode()))
    path = os.path.join(TMP, 'synthetic.pcap')
    wrpcap(path, pkts)

    rx1_s, rx2_s, rx1_g, rx2_g = read_pcap(path, validate_checksums=True)
    rx1 = build_epoch_list(rx1_s, rx1_g)
    rx2 = build_epoch_list(rx2_s, rx2_g)
    assert len(rx1) == 130 and len(rx2) == 130

    feats = extract_features(rx1, rx2, profile='full')
    expected = set(feature_names('full'))
    assert set(feats) == expected, set(feats) ^ expected
    assert len(expected) == 138, len(expected)

    # inter-receiver distance should start near 4 m and grow with drift
    assert 3.0 < feats['inter_rx_distance_m_min'] < 5.0
    assert feats['inter_rx_distance_m_max'] > 6.0
    assert feats['mean_snr_mean'] > 35

    dep = extract_features(rx1, rx2, profile='deployable')
    assert len(dep) == len(feature_names('deployable')) == 93
    assert not any(k.startswith(('mean_snr', 'pdop', 'vdop', 'fix_mode', 'hdop_gsa')) for k in dep)

    frame = epochs_to_frame(rx1, rx2)
    assert frame.shape[0] == 260
    pq = os.path.join(TMP, 'epochs.parquet')
    frame.to_parquet(pq, index=False)
    print(f'pcap pipeline OK (full=138 feats, deployable={len(dep)}, '
          f'inter_rx max={feats["inter_rx_distance_m_max"]:.2f} m, '
          f'epoch frame {frame.shape})')


def test_scs_pipeline():
    scs_path = os.path.join(TMP, 'GPGGA_001.Raw')
    iso_path = os.path.join(TMP, 'cnav_iso.log')
    lat, lon = 40.9, 28.9
    with open(scs_path, 'w') as scs, open(iso_path, 'w') as iso:
        for i in range(400):
            t = 12 * 3600 + i
            hh, mm, ss = int(t // 3600), int((t % 3600) // 60), int(t % 60)
            sents = make_sentences(t, lat + i * 4e-5, lon + i * 4e-5)
            for s in (sents[1], sents[5]):  # GGA + VTG only (NOAA reality)
                scs.write(f'05/28/2018,{hh:02d}:{mm:02d}:{ss:02d}.100,{s}\n')
                iso.write(f'2018-05-28T{hh:02d}:{mm:02d}:{ss:02d}.100Z {s}\n')
        scs.write('garbage line, not nmea\n')
        scs.write('05/28/2018,12:06:41.100,$GPGGA,truncat\n')  # bad checksumless stub

    ext, payload = split_log_line('05/28/2018,12:00:00.100,$GPGGA,x')
    assert payload.startswith('$GPGGA') and ext > 1.5e9
    ext2, _ = split_log_line('2018-05-28T12:00:00.100Z $GPGGA,x')
    assert abs(ext - ext2) < 1e-6

    for path in (scs_path, iso_path):
        sentences, gsv = read_scs_files([path])
        epochs = build_epoch_list(sentences, gsv)
        assert len(epochs) == 400, len(epochs)
        # canonical sog/cog must be filled from VTG (no RMC in the file)
        assert abs(epochs[10]['sog_knots'] - 10.0) < 1e-6
        assert abs(epochs[10]['cog_deg'] - 45.0) < 1e-6

        wins = list(sliding_windows(epochs, window_s=120, stride_s=60, min_epochs=60))
        assert len(wins) == 5, len(wins)
        feats = extract_features(wins[0][1], [], profile='deployable')
        assert not math.isnan(feats['sog_knots_mean'])
        assert math.isnan(feats['inter_rx_distance_m_mean'])  # single device
    print(f'scs pipeline OK (400 epochs, {len(wins)} windows, both prefix styles)')


def test_splits():
    import pandas as pd
    from mspoof import splits as sp
    rng = np.random.default_rng(0)
    rows = []
    speeds = list(range(0, 76, 4))
    for scen in ('A1', 'A3'):
        for v in speeds:
            for rep in range(20):
                for label in ('spoofed', 'unspoofed'):
                    rows.append({'scenario': scen, 'label': label, 'index': rep,
                                 'param_1_name': 'shift_angle' if scen == 'A3' else 'distance_to_ship',
                                 'param_1_value': rng.integers(0, 180),
                                 'param_2_name': 'shift_speed' if scen == 'A3' else 'time_difference',
                                 'param_2_value': v})
    df = pd.DataFrame(rows)

    tr, te = sp.repetition_split(df)
    assert tr.sum() + te.sum() == len(df) and te.sum() == len(df) * 4 // 20

    folds = list(sp.kfold_repetition(df))
    assert len(folds) == 5
    assert sum(m.sum() for _, _, m in folds) == len(df)

    tr, te = sp.param_holdout(df, **sp.PARAM_HOLDOUT_PRESETS['a3_speed_low'])
    assert not (tr & te).any()
    a3 = df['scenario'] == 'A3'
    assert te.sum() == (a3 & (df['param_2_value'] <= 16)).sum()
    assert tr.sum() == (a3 & (df['param_2_value'] > 16)).sum()
    # both labels present on both sides (paired baselines move together)
    assert set(df.loc[te, 'label'].unique()) == {'spoofed', 'unspoofed'}
    assert not df.loc[tr | te, 'scenario'].eq('A1').any()

    tr2, _ = sp.param_holdout(df, other_scenarios='train',
                              **sp.PARAM_HOLDOUT_PRESETS['a3_speed_low'])
    assert df.loc[tr2, 'scenario'].eq('A1').any()
    print('splits OK')


if __name__ == '__main__':
    import shutil
    try:
        test_geo()
        test_nmea()
        test_pcap_pipeline()
        test_scs_pipeline()
        test_splits()
        print('\nALL TESTS PASSED')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
