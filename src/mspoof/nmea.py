"""NMEA 0183 sentence parsing.

Design notes for the journal-version refactor:

* Parsers are keyed by the 3-letter sentence formatter (RMC, GGA, ...), not by
  the full ``$GPxxx`` header. Real receivers emit GN/GL/GA/GB talker prefixes
  alongside GP; MARSIM emits GP only. Matching on the formatter makes the same
  code path work for both.
* Checksums are validated when present (``verify_checksum``). MARSIM data is
  clean, but real SCS/R2R logs occasionally contain truncated lines; those are
  dropped rather than half-parsed.
* Every field accessor is NaN-safe.
"""

import math

_FORMATTERS = ('RMC', 'GGA', 'GSA', 'GSV', 'VTG', 'GLL', 'ZDA')


def safe_float(val):
    if val is None:
        return float('nan')
    if isinstance(val, str) and val.strip() == '':
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def nmea_time_to_seconds(time_str):
    """HHMMSS.sss UTC field to seconds since midnight."""
    if not time_str or not str(time_str).strip():
        return float('nan')
    try:
        t = float(time_str)
    except (ValueError, TypeError):
        return float('nan')
    hours = int(t / 10000)
    minutes = int((t % 10000) / 100)
    seconds = t % 100
    return hours * 3600 + minutes * 60 + seconds


def nmea_coord_to_decimal(coord_str, hemisphere):
    """(D)DDMM.MMMM + hemisphere to signed decimal degrees."""
    if not coord_str or not str(coord_str).strip():
        return float('nan')
    try:
        val = float(coord_str)
    except (ValueError, TypeError):
        return float('nan')
    degrees = int(val / 100)
    minutes = val - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere in ('S', 'W'):
        decimal = -decimal
    return decimal


def verify_checksum(sentence):
    """Return True if the sentence has a valid checksum or none at all."""
    if '*' not in sentence:
        return True
    body, _, cs = sentence.rpartition('*')
    body = body.lstrip('$!')
    try:
        expected = int(cs.strip()[:2], 16)
    except ValueError:
        return False
    actual = 0
    for ch in body:
        actual ^= ord(ch)
    return actual == expected


def split_sentence(sentence):
    """Split an NMEA sentence into fields, stripping the checksum suffix."""
    if '*' in sentence:
        sentence = sentence.rpartition('*')[0]
    return sentence.split(',')


def formatter_of(sentence):
    """Return the 3-letter formatter of a sentence, or None.

    ``$GPGGA,...`` -> ``GGA``; ``$GNRMC,...`` -> ``RMC``; non-NMEA -> None.
    Proprietary sentences ($P...) return None.
    """
    if not sentence or sentence[0] not in '$!':
        return None
    header = sentence[1:].split(',', 1)[0]
    if len(header) < 5 or header.startswith('P'):
        return None
    fmt = header[-3:]
    return fmt if fmt in _FORMATTERS else None


# ---------------------------------------------------------------------------
# Per-formatter parsers. Each takes the comma-split fields (checksum removed)
# and returns a flat dict of canonical keys.
# ---------------------------------------------------------------------------

def parse_rmc(fields):
    """$--RMC,UTC,Status,Lat,N/S,Lon,E/W,SOG,COG,Date,MagVar,E/W[,Mode]"""
    out = {}
    if len(fields) < 9:
        return out
    out['utc_time'] = nmea_time_to_seconds(fields[1])
    out['lat_deg'] = nmea_coord_to_decimal(fields[3], fields[4]) if len(fields) > 4 else float('nan')
    out['lon_deg'] = nmea_coord_to_decimal(fields[5], fields[6]) if len(fields) > 6 else float('nan')
    out['sog_rmc_knots'] = safe_float(fields[7])
    out['cog_rmc_deg'] = safe_float(fields[8])
    return out


def parse_gga(fields):
    """$--GGA,UTC,Lat,N/S,Lon,E/W,Quality,NumSats,HDOP,Alt,M,Geoid,M,Age,RefID"""
    out = {}
    if len(fields) < 10:
        return out
    out['utc_time'] = nmea_time_to_seconds(fields[1])
    out['lat_deg'] = nmea_coord_to_decimal(fields[2], fields[3]) if len(fields) > 3 else float('nan')
    out['lon_deg'] = nmea_coord_to_decimal(fields[4], fields[5]) if len(fields) > 5 else float('nan')
    out['fix_quality'] = safe_float(fields[6])
    out['num_satellites'] = safe_float(fields[7])
    out['hdop'] = safe_float(fields[8])
    out['altitude'] = safe_float(fields[9])
    return out


def parse_gsa(fields):
    """$--GSA,Mode1,Mode2,SV1..SV12,PDOP,HDOP,VDOP"""
    out = {}
    if len(fields) < 18:
        return out
    out['fix_mode'] = safe_float(fields[2])
    out['pdop'] = safe_float(fields[15])
    out['hdop_gsa'] = safe_float(fields[16])
    out['vdop'] = safe_float(fields[17])
    return out


def parse_gsv(fields):
    """$--GSV,NumMsg,MsgNum,NumSV,[PRN,Elev,Azim,SNR]x4 -> list of SNR values."""
    snrs = []
    if len(fields) < 4:
        return snrs
    idx = 4
    while idx + 3 < len(fields):
        snr = safe_float(fields[idx + 3])
        if not math.isnan(snr):
            snrs.append(snr)
        idx += 4
    return snrs


def parse_vtg(fields):
    """$--VTG,COG_True,T,COG_Mag,M,SOG_kn,N,SOG_kmh,K[,Mode]"""
    out = {}
    if len(fields) < 8:
        return out
    out['cog_true'] = safe_float(fields[1])
    out['cog_magnetic'] = safe_float(fields[3])
    out['sog_vtg_knots'] = safe_float(fields[5])
    out['sog_vtg_kmh'] = safe_float(fields[7])
    return out


PARSERS = {
    'RMC': parse_rmc,
    'GGA': parse_gga,
    'GSA': parse_gsa,
    'VTG': parse_vtg,
    # GSV handled separately (returns a list, accumulates per epoch)
    # GLL skipped: redundant with GGA/RMC position
    # ZDA consumed by readers for date context only
}

# Formatters that carry their own UTC field and therefore OPEN an epoch.
EPOCH_ANCHORS = ('RMC', 'GGA')
