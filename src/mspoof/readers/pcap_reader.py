"""MARSIM pcap reader.

Produces the reader-standard structures consumed by epochs.py:
    (rx1_sentences, rx2_sentences, rx1_gsv, rx2_gsv)
where sentences maps utc_seconds -> merged field dict and gsv maps
utc_seconds -> list of SNR values.

Receiver identification is by (source IP, source UDP port), configurable so
the same reader works if MARSIM regenerates with different addressing.
"""

import math
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ.setdefault('SCAPY_USE_LIBPCAP', '0')

from scapy.all import PcapReader, IP, UDP, Raw  # noqa: E402

from .. import nmea  # noqa: E402

DEFAULT_RECEIVERS = {
    ('192.168.0.10', 62996): 1,
    ('192.168.0.11', 62997): 2,
}


def read_pcap(filepath, receivers=None, validate_checksums=False):
    """Parse one MARSIM pcap into per-receiver epoch dicts.

    validate_checksums defaults to False for MARSIM (clean data, and skipping
    validation keeps extraction throughput identical to the original code).
    """
    receivers = receivers or DEFAULT_RECEIVERS
    sentences = {1: defaultdict(dict), 2: defaultdict(dict)}
    gsv = {1: defaultdict(list), 2: defaultdict(list)}
    current_epoch = {1: None, 2: None}

    try:
        reader = PcapReader(str(filepath))
    except Exception:
        return sentences[1], sentences[2], gsv[1], gsv[2]

    try:
        for pkt in reader:
            if not (pkt.haslayer(IP) and pkt.haslayer(UDP) and pkt.haslayer(Raw)):
                continue
            rx_id = receivers.get((pkt[IP].src, pkt[UDP].sport))
            if rx_id is None:
                continue
            try:
                payload = pkt[Raw].load.decode('ascii', errors='ignore').strip()
            except Exception:
                continue

            fmt = nmea.formatter_of(payload)
            if fmt is None:
                continue
            if validate_checksums and not nmea.verify_checksum(payload):
                continue

            fields = nmea.split_sentence(payload)

            if fmt in nmea.EPOCH_ANCHORS:
                parsed = nmea.PARSERS[fmt](fields)
                t = parsed.get('utc_time', float('nan'))
                if not math.isnan(t):
                    current_epoch[rx_id] = t
                    sentences[rx_id][t].update(parsed)
            elif fmt == 'GSV':
                t = current_epoch[rx_id]
                if t is not None:
                    gsv[rx_id][t].extend(nmea.parse_gsv(fields))
            elif fmt in nmea.PARSERS:  # GSA, VTG
                t = current_epoch[rx_id]
                if t is not None:
                    sentences[rx_id][t].update(nmea.PARSERS[fmt](fields))
            # GLL/ZDA intentionally skipped for pcap input

    except Exception:
        pass
    finally:
        reader.close()

    return sentences[1], sentences[2], gsv[1], gsv[2]
