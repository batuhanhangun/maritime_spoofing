"""Reader for real vessel navigation logs (NOAA SCS, R2R, OpenRVDAS/LDS).

Verified against the archive formats of the two public sources selected for
the journal-version validation experiment:

* NOAA OMAO-SCS accessions at NCEI (e.g. Okeanos Explorer EX-23-03,
  accession 0279574): SCS ``.Raw`` files, one sentence family per file,
  each line prefixed with an external-clock timestamp:
      ``05/28/2018,00:00:01.234,$GPGGA,...*5C``
* R2R academic-fleet GNSS filesets (e.g. Falkor FK180528, fileset 129529,
  format "DAS: NOAA SCS: external clock + NMEA GGA") use the same SCS line
  format. OpenRVDAS/LDS-era vessels instead prefix an ISO timestamp:
      ``2018-05-28T00:00:01.234Z $GPGGA,...*5C``

The reader sniffs the prefix style per line, so mixed archives work. Epochs
are keyed by the NMEA UTC field when the sentence carries one (GGA/RMC) and
by the external clock otherwise, mirroring the pcap reader's anchor logic.

Key differences from MARSIM handled here:
* Receiver identity comes from the device/file, not the packet source. The
  caller assigns rx_id per file (e.g. C-Nav log -> 1, Seapath log -> 2).
* Real cruises span days; utc_time is seconds since midnight plus a day
  offset inferred from the external clock, so multi-day logs stay monotonic.
* Checksums are validated by default; truncated serial lines are common.
"""

import math
import re
from collections import defaultdict
from datetime import datetime, timezone

from .. import nmea

# ``05/28/2018,00:00:01.234,$GPGGA,...``
_SCS_RE = re.compile(
    r'^(?P<date>\d{2}/\d{2}/\d{4}),(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?),(?P<nmea>[$!].*)$')
# ``2018-05-28T00:00:01.234Z $GPGGA,...`` (also tolerates comma separator)
_ISO_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)[,\s]+(?P<nmea>[$!].*)$')


def _parse_scs_clock(date_str, time_str):
    dt = datetime.strptime(f'{date_str} {time_str.split(".")[0]}', '%m/%d/%Y %H:%M:%S')
    frac = 0.0
    if '.' in time_str:
        frac = float('0.' + time_str.split('.', 1)[1])
    return dt.replace(tzinfo=timezone.utc).timestamp() + frac


def _parse_iso_clock(ts):
    ts = ts.replace(' ', 'T').rstrip('Z')
    base, frac = (ts.split('.', 1) + ['0'])[:2]
    dt = datetime.strptime(base, '%Y-%m-%dT%H:%M:%S')
    return dt.replace(tzinfo=timezone.utc).timestamp() + float('0.' + frac)


def split_log_line(line):
    """Return (external_epoch_seconds, nmea_sentence) or (None, None)."""
    line = line.strip()
    if not line:
        return None, None
    m = _SCS_RE.match(line)
    if m:
        try:
            return _parse_scs_clock(m.group('date'), m.group('time')), m.group('nmea')
        except ValueError:
            return None, None
    m = _ISO_RE.match(line)
    if m:
        try:
            return _parse_iso_clock(m.group('ts')), m.group('nmea')
        except ValueError:
            return None, None
    # Bare NMEA with no prefix (some archives): still usable if it anchors.
    if line[0] in '$!':
        return float('nan'), line
    return None, None


def _absolute_utc(nmea_utc, ext_clock):
    """Combine seconds-since-midnight from the sentence with the external
    clock's day, producing an absolute, monotonic timestamp in seconds.

    Handles the receiver clock and logging clock straddling midnight by
    choosing the day offset that minimizes |nmea_abs - ext_clock|.
    """
    if math.isnan(nmea_utc):
        return ext_clock
    if ext_clock is None or math.isnan(ext_clock):
        return nmea_utc
    day = math.floor(ext_clock / 86400.0) * 86400.0
    candidates = [day + nmea_utc - 86400.0, day + nmea_utc, day + nmea_utc + 86400.0]
    return min(candidates, key=lambda c: abs(c - ext_clock))


def read_scs_files(filepaths, validate_checksums=True):
    """Parse one receiver's log file(s) into (sentences, gsv) epoch dicts.

    filepaths: iterable of paths belonging to ONE device (SCS splits sentence
    families across files and days; pass them all together).
    """
    sentences = defaultdict(dict)
    gsv = defaultdict(list)
    current_epoch = None

    for path in filepaths:
        with open(path, 'r', errors='ignore') as fh:
            for line in fh:
                ext_clock, payload = split_log_line(line)
                if payload is None:
                    continue
                fmt = nmea.formatter_of(payload)
                if fmt is None:
                    continue
                if validate_checksums and not nmea.verify_checksum(payload):
                    continue
                fields = nmea.split_sentence(payload)

                if fmt in nmea.EPOCH_ANCHORS:
                    parsed = nmea.PARSERS[fmt](fields)
                    t_rel = parsed.get('utc_time', float('nan'))
                    if math.isnan(t_rel):
                        continue
                    t_abs = _absolute_utc(t_rel, ext_clock)
                    parsed['utc_time'] = t_abs
                    current_epoch = t_abs
                    sentences[t_abs].update(parsed)
                elif fmt == 'GSV':
                    if current_epoch is not None:
                        gsv[current_epoch].extend(nmea.parse_gsv(fields))
                elif fmt in nmea.PARSERS:
                    if current_epoch is not None:
                        sentences[current_epoch].update(nmea.PARSERS[fmt](fields))

    return sentences, gsv


def sliding_windows(epochs, window_s=120.0, stride_s=60.0, min_epochs=60):
    """Yield (window_start, epoch_sublist) over a sorted epoch list.

    Mirrors the MARSIM recording length (120 s) so that MARSIM-trained models
    see windows with the same aggregation support. min_epochs guards against
    logging gaps producing degenerate feature vectors.
    """
    if not epochs:
        return
    times = [e['utc_time'] for e in epochs]
    t0, t_end = times[0], times[-1]
    start = t0
    i = 0
    while start + window_s <= t_end + 1e-9:
        while i < len(times) and times[i] < start:
            i += 1
        j = i
        while j < len(times) and times[j] < start + window_s:
            j += 1
        window = epochs[i:j]
        if len(window) >= min_epochs:
            yield start, window
        start += stride_s
