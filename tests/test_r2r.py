"""Fixture test for the R2R split-family reader, replicating the FK180528
formats observed in the wild: SCS clock prefix, per-family daily files,
GNGGA talker, 2 Hz C-Nav vs 1 Hz Seapath at ~0.55 s phase, _2 duplicates."""
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mspoof.readers.r2r_reader import discover_device_files, read_r2r_device
from mspoof.epochs import build_epoch_list, epochs_to_frame
from mspoof.features import extract_features
from mspoof import nmea

TMP = os.path.join(os.path.dirname(__file__), '_tmp_r2r')

LAT0, LON0 = 31.19873, -123.98936          # ~FK180528 area
M_PER_DEG = 111320.0
SOG_KN, COG = 12.0, 0.0                     # northbound
LAG = 0.31                                  # logger latency seconds


def _chk(body):
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f'${body}*{c:02X}'


def _fmt_lat(lat):
    d = int(abs(lat)); m = (abs(lat) - d) * 60
    return f'{d:02d}{m:09.6f}', 'N' if lat >= 0 else 'S'


def _fmt_lon(lon):
    d = int(abs(lon)); m = (abs(lon) - d) * 60
    return f'{d:03d}{m:09.6f}', 'E' if lon >= 0 else 'W'


def _pos(t, lat_off_m=0.0):
    lat = LAT0 + (SOG_KN * 0.514444 * t) / M_PER_DEG + lat_off_m / M_PER_DEG
    return lat, LON0


def _utc_str(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f'{int(h):02d}{int(m):02d}{s:05.2f}'


def _clock(t):
    h, r = divmod(t + LAG, 3600); m, s = divmod(r, 60)
    return f'06/09/2018,{int(h):02d}:{int(m):02d}:{s:06.3f}'


def _gga(talker, t, lat_off):
    lat, lon = _pos(t, lat_off)
    la, ns = _fmt_lat(lat); lo, ew = _fmt_lon(lon)
    body = (f'{talker}GGA,{_utc_str(t)},{la},{ns},{lo},{ew},2,15,0.6,'
            f'-21.350,M,0.000,M,5.0,0436')
    return f'{_clock(t)},{_chk(body)}'


def _rmc(talker, t, lat_off):
    lat, lon = _pos(t, lat_off)
    la, ns = _fmt_lat(lat); lo, ew = _fmt_lon(lon)
    body = (f'{talker}RMC,{_utc_str(t)},A,{la},{ns},{lo},{ew},'
            f'{SOG_KN:.1f},{COG:.1f},090618,,,D')
    return f'{_clock(t)},{_chk(body)}'


def _vtg(talker, t):
    body = (f'{talker}VTG,{COG:.1f},T,,M,{SOG_KN:.1f},N,'
            f'{SOG_KN*1.852:.1f},K,D')
    return f'{_clock(t)},{_chk(body)}'


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def build_fixture(n_sec=400):
    shutil.rmtree(TMP, ignore_errors=True)
    cnav = os.path.join(TMP, 'gnss')
    spath = os.path.join(TMP, 'ins', 'data')
    # C-Nav: GGA at 2 Hz, RMC + VTG at 1 Hz, integer-second phase
    gga = [_gga('GN', k / 2.0, 0.0) for k in range(n_sec * 2)]
    rmc = [_rmc('GN', float(k), 0.0) for k in range(n_sec)]
    vtg = [_vtg('GN', float(k)) for k in range(n_sec)]
    _write(os.path.join(cnav, 'COM18-CNAV-GGA-RAW_20180609-000001.raw'), gga)
    _write(os.path.join(cnav, 'COM18-CNAV-GGA-RAW_20180609-000001_2.raw'), gga)
    _write(os.path.join(cnav, 'COM18-CNAV-RMC-RAW_20180609-000001.raw'), rmc)
    _write(os.path.join(cnav, 'COM18-CNAV-RMC-RAW_20180609-000001_2.raw'), rmc)
    _write(os.path.join(cnav, 'COM18-CNAV-VTG-RAW_20180609-000001.raw'), vtg)
    # decoys that must be ignored
    _write(os.path.join(cnav, 'COM18-CNAV-GLL-RAW_20180609-000001.raw'),
           ['06/09/2018,00:00:00.500,$GNGLL,junk*00'])
    # Seapath: 1 Hz at 0.55 s phase, antenna 30 m north of C-Nav
    g2 = [_gga('GP', k + 0.55, 30.0) for k in range(n_sec)]
    r2 = [_rmc('GP', k + 0.55, 30.0) for k in range(n_sec)]
    v2 = [_vtg('GP', k + 0.55) for k in range(n_sec)]
    _write(os.path.join(spath, 'COM12-Seapath-GGA-RAW_20180609-000001.Raw'), g2)
    _write(os.path.join(spath, 'COM12-Seapath-RMC-RAW_20180609-000001.Raw'), r2)
    _write(os.path.join(spath, 'COM12-Seapath-VTG-RAW_20180609-000001.Raw'), v2)
    return cnav, os.path.join(TMP, 'ins')


def test_r2r():
    cnav, ins = build_fixture()

    fam = discover_device_files(cnav)
    assert set(fam) == {'GGA', 'RMC', 'VTG'}, fam
    assert all(len(v) == 1 for v in fam.values()), 'dedupe failed'
    assert not fam['GGA'][0].endswith('_2.raw'), 'kept duplicate copy'

    st1, st2 = {}, {}
    s1, g1 = read_r2r_device(cnav, stats_out=st1)
    s2, g2 = read_r2r_device(ins, stats_out=st2)

    # 2 Hz decimated to 1 Hz; phases preserved
    assert 395 <= len(s1) <= 400, len(s1)
    assert 395 <= len(s2) <= 400, len(s2)
    fracs1 = {round(t % 1.0, 2) for t in s1}
    fracs2 = {round(t % 1.0, 2) for t in s2}
    assert fracs1 == {0.0}, fracs1
    assert fracs2 == {0.55}, fracs2

    # logger-latency estimate and VTG attachment
    assert abs(st1['lag_s'] - LAG) < 0.02, st1
    e1 = build_epoch_list(s1, g1)
    e2 = build_epoch_list(s2, g2)
    cog_cov = sum(1 for e in e1
                  if not math.isnan(e.get('cog_true', float('nan')))) / len(e1)
    assert cog_cov > 0.95, f'VTG attachment coverage {cog_cov:.2f}'

    from mspoof.epochs import compute_inter_receiver
    import statistics
    inter = compute_inter_receiver(e1, e2)  # dict of key -> per-epoch list
    dists = [v for v in inter['inter_rx_distance_m'] if not math.isnan(v)]
    assert len(dists) > 350, 'inter-receiver matching failed across 0.55 s phase'
    med = statistics.median(dists)
    assert abs(med - 30.0) < 1.5, f'baseline {med:.2f} m'

    feats = extract_features(e1, e2, profile='deployable')
    bad = [k for k, v in feats.items()
           if isinstance(v, float) and math.isnan(v)
           and not k.startswith(('cog_magnetic',))]  # empty in real VTG
    assert not bad, f'NaN features: {bad[:8]}'
    print(f'r2r OK (epochs {len(s1)}/{len(s2)}, lag {st1["lag_s"]:.3f}s, '
          f'cog coverage {cog_cov:.2f}, inter_rx {med:.2f} m, '
          f'{len(feats)} feats)')


if __name__ == '__main__':
    try:
        test_r2r()
        print('ALL R2R TESTS PASSED')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
